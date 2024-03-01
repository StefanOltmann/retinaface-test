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

import ai.djl.ModelException
import ai.djl.inference.Predictor
import ai.djl.modality.cv.Image
import ai.djl.modality.cv.ImageFactory
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject
import ai.djl.repository.zoo.Criteria
import ai.djl.training.util.ProgressBar
import ai.djl.translate.TranslateException
import java.io.IOException
import kotlin.io.path.Path

object RetinaFaceDetection {

    init {
        println("Loaded RetinaFaceDetection")
    }

    @Suppress("MagicNumber")
    val translator = FaceDetectionTranslator(
        confThresh = 0.85,
        nmsThresh = 0.45,
        variance = doubleArrayOf(0.1, 0.2),
        topK = 5000,
        scales = arrayOf(intArrayOf(16, 32), intArrayOf(64, 128), intArrayOf(256, 512)),
        steps = intArrayOf(8, 16, 32)
    )

    val criteria =
        Criteria.builder()
            .setTypes(Image::class.java, DetectedObjects::class.java)
            .optModelUrls("jar:/META-INF/models/retinaface.zip")
            // .optModelUrls("https://resources.djl.ai/test-models/pytorch/retinaface.zip")
            .optModelName("retinaface")
            .optTranslator(translator)
            .optProgress(ProgressBar())
            .optEngine("PyTorch")
            .build()

    var predictor: Predictor<Image, DetectedObjects>? = null

    @Throws(IOException::class, ModelException::class, TranslateException::class)
    fun predict(pathString: String): Set<BoundingBox> {

        val boundingBoxes = mutableSetOf<BoundingBox>()

        /* Only load the model once. */
        if (predictor == null) {

            synchronized(this) {

                if (predictor == null) {

                    val model = criteria.loadModel()
                    predictor = model.newPredictor()
                }
            }
        }

        val path = Path(pathString)

        val start = System.currentTimeMillis()

        val img = ImageFactory.getInstance().fromFile(path)

        val startDetection = System.currentTimeMillis()

        val detection: DetectedObjects = predictor!!.predict(img)

        val durationOverall = System.currentTimeMillis() - start
        val durationDetection = System.currentTimeMillis() - startDetection

        if (detection.numberOfObjects == 0)
            return emptySet()

        println("Detection for $path took $durationOverall ms with $durationDetection for detection.")

        for (detectedObject in detection.items<DetectedObject>()) {

            val box = detectedObject.boundingBox

            boundingBoxes.add(
                BoundingBox(
                    centerX = box.bounds.x,
                    centerY = box.bounds.y,
                    width = box.bounds.width,
                    height = box.bounds.height
                )
            )
        }

        return boundingBoxes
    }
}
