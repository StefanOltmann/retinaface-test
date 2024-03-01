package de.stefan_oltmann

import RetinaFaceDetection
import java.io.File

fun main() {

    val files = File("testfiles").listFiles()

    for (file in files) {

        val boxes = RetinaFaceDetection.predict(file.absolutePath)

        println("${file.name} = $boxes")
    }
}