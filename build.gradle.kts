plugins {
    kotlin("jvm") version "1.9.21"
}

group = "de.stefan_oltmann"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven("https://oss.sonatype.org/content/repositories/snapshots/")
}

dependencies {

    val osName = System.getProperty("os.name")
    val targetOs = when {
        osName == "Mac OS X" -> "macos"
        osName.startsWith("Win") -> "windows"
        osName.startsWith("Linux") -> "linux"
        else -> error("Unsupported OS: $osName")
    }

    val djlVersion = "0.27.0-SNAPSHOT"
    val pytorchVersion = "2.1.1"

    implementation("ai.djl.pytorch:pytorch-engine:$djlVersion") {
        exclude(group = "org.apache.commons", module = "commons-compress")
    }

    implementation("ai.djl.pytorch:pytorch-jni:$pytorchVersion-$djlVersion")

    if (targetOs == "windows")
        implementation("ai.djl.pytorch:pytorch-native-cpu:$pytorchVersion:win-x86_64")

    if (targetOs == "macos")
        implementation("ai.djl.pytorch:pytorch-native-cpu:$pytorchVersion:osx-aarch64")
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(17)
}