/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

import ai.djl.modality.cv.Image
import ai.djl.modality.cv.output.BoundingBox
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.Landmark
import ai.djl.modality.cv.output.Rectangle
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.ceil

class FaceDetectionTranslator(
    private val confThresh: Double,
    private val nmsThresh: Double,
    private val variance: DoubleArray,
    private val topK: Int,
    private val scales: Array<IntArray>,
    private val steps: IntArray
) : Translator<Image, DetectedObjects> {

    private var width = 0
    private var height = 0

    override fun processInput(
        ctx: TranslatorContext,
        input: Image
    ): NDList {

        width = input.width
        height = input.height

        var array = input.toNDArray(ctx.ndManager, Image.Flag.COLOR)
        array = array.transpose(2, 0, 1).flip(0) // HWC -> CHW RGB -> BGR

        // The network by default takes float32
        if (array.dataType != DataType.FLOAT32)
            array = array.toType(DataType.FLOAT32, false)

        val mean = ctx.ndManager.create(floatArrayOf(104f, 117f, 123f), Shape(3, 1, 1))

        array = array.sub(mean)

        return NDList(array)
    }

    override fun processOutput(ctx: TranslatorContext, list: NDList): DetectedObjects {

        val manager = ctx.ndManager
        val scaleXY = variance[0]
        val scaleWH = variance[1]

        var prob = list[1][":, 1:"]
        prob = NDArrays.stack(
            NDList(
                prob.argMax(1).toType(DataType.FLOAT32, false),
                prob.max(intArrayOf(1))
            )
        )

        val boxRecover = boxRecover(manager, width, height, scales, steps)

        var boundingBoxes = list[0]

        val bbWH = boundingBoxes[":, 2:"]
            .mul(scaleWH)
            .exp()
            .mul(boxRecover[":, 2:"])

        val bbXY = boundingBoxes[":, :2"]
            .mul(scaleXY)
            .mul(boxRecover[":, 2:"])
            .add(boxRecover[":, :2"])
            .sub(bbWH.mul(0.5f))

        boundingBoxes = NDArrays.concat(NDList(bbXY, bbWH), 1)

        // filter the result below the threshold
        val cutOff = prob[1].gt(confThresh)
        boundingBoxes = boundingBoxes.transpose().booleanMask(cutOff, 1).transpose()
        prob = prob.booleanMask(cutOff, 1)

        // start categorical filtering
        val order = prob[1].argSort()[":$topK"].toLongArray()
        prob = prob.transpose()

        val classNames: MutableList<String> = ArrayList()
        val probabilities: MutableList<Double> = ArrayList()
        val allBoundingBoxes: MutableList<BoundingBox> = ArrayList()

        val recorder: MutableMap<Int, MutableList<BoundingBox>> = ConcurrentHashMap()

        for (index in order.indices.reversed()) {

            val currMaxLoc = order[index]
            val classProb = prob[currMaxLoc].toFloatArray()
            val classId = classProb[0].toInt()
            val probability = classProb[1].toDouble()

            val boxArray = boundingBoxes[currMaxLoc].toDoubleArray()

            val rectangle = Rectangle(boxArray[0], boxArray[1], boxArray[2], boxArray[3])

            val boxes = recorder.getOrDefault(classId, ArrayList())

            var belowIoU = true

            for (box in boxes) {
                if (box.getIoU(rectangle) > nmsThresh) {
                    belowIoU = false
                    break
                }
            }

            if (belowIoU) {

                val landmark = Landmark(boxArray[0], boxArray[1], boxArray[2], boxArray[3], emptyList())

                boxes.add(landmark)

                recorder[classId] = boxes

                val className = "Face"
                classNames.add(className)

                probabilities.add(probability)

                allBoundingBoxes.add(landmark)
            }
        }

        return DetectedObjects(classNames, probabilities, allBoundingBoxes)
    }

    private fun boxRecover(
        manager: NDManager,
        width: Int,
        height: Int,
        scales: Array<IntArray>,
        steps: IntArray
    ): NDArray {

        val aspectRatio = Array(steps.size) { IntArray(2) }

        for (i in steps.indices) {
            val wRatio = ceil((width.toFloat() / steps[i]).toDouble()).toInt()
            val hRatio = ceil((height.toFloat() / steps[i]).toDouble()).toInt()
            aspectRatio[i] = intArrayOf(hRatio, wRatio)
        }

        val defaultBoxes: MutableList<DoubleArray> = ArrayList()

        for (idx in steps.indices) {
            val scale = scales[idx]
            for (h in 0 until aspectRatio[idx][0]) {
                for (w in 0 until aspectRatio[idx][1]) {
                    for (i in scale) {
                        val skx = i * 1.0 / width
                        val sky = i * 1.0 / height
                        val cx = (w + 0.5) * steps[idx] / width
                        val cy = (h + 0.5) * steps[idx] / height
                        defaultBoxes.add(doubleArrayOf(cx, cy, skx, sky))
                    }
                }
            }
        }

        val boxes = Array(defaultBoxes.size) { DoubleArray(defaultBoxes[0].size) }

        for (i in defaultBoxes.indices)
            boxes[i] = defaultBoxes[i]

        return manager.create(boxes).clip(0.0, 1.0)
    }
}
