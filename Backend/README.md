# Flui3d Chat Backend

This folder contains the source code for the Flui3d Chat's backend, a Spring Boot application that powers the LLM-Based Online Design Platform for Microfluidics. It serves as an intermediary between the frontend user interface and the core services, including the LLM for design generation, the validation and correction logic, and the synthesis tool for generating chip layouts.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:
* **Java Development Kit (JDK)**: Version **21** or higher.
* **Apache Maven**: To build the project and manage dependencies.
* **Git**: To clone the required repositories.
* **Gurobi Solver**: The design synthesis tool requires a valid Gurobi license and installation to solve the design synthesis optimization problem.

---

## ⚙️ Setup Instructions

Follow these steps to set up the backend and its dependencies.

### 1. Clone the Backend Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/TUM-EDA/Flui3d-Chat.git
cd Flui3d-Chat/Backend
```

### 2. Install the Design Synthesis Dependency
The backend relies on a modified version of the `3M-DeSyn` tool for chip synthesis. You need to install it into your local Maven repository.


i. **Install the JAR to your local Maven repository:**
Navigate to the `Backend/designsynthesis` directory and run the following command.

> **Note:** Adjust the `-Dfile` path if your directory structure is different. The path should point to the JAR file created in the previous step.

```bash
./mvnw install:install-file -Dfile=designsynthesis/3DMF-DesignSynthesis-1.0-SNAPSHOT.jar -DgroupId=com.threedmf -DartifactId=designsynthesis -Dversion=1.0-SNAPSHOT -Dpackaging=jar
```

### 3. Set Up Dependent Services
The backend communicates with two other essential services that must be running:

* **Ollama (Modified Fork)**: This project uses a custom build of Ollama to handle the LLM requests with specific grammar constraints. Clone and run it by following the instructions at:
    * [Customized Ollama](../Customized%20Ollama/README.md)
    * Ensure the models specified in `application.properties` (e.g., `microfluidic_llama_3`, `microfluidic_qwen3_reasoning`) are pulled and available in your Ollama instance.

* **Frontend**: This is the Vue.js single-page application that provides the user interface. Clone and run it by following the instructions at:
    * [Frontend](../Frontend/README.md)

---

## 🔧 Configuration

The primary configuration is managed in the `src/main/resources/application.properties` file. Make sure the values match your setup.

```properties
# Spring Boot application settings
spring.application.name=flui3d-chat-backend
server.port=8090

# URL for the locally running Ollama instance
ollama.api.url=http://127.0.0.1:11434

# Names of the LLMs hosted in Ollama
ollama.modelname.baseline=microfluidic_llama_3
ollama.modelname.reasoning=microfluidic_qwen3_reasoning

# Allowed origins for Cross-Origin Resource Sharing (CORS)
# Should match the URL of the running frontend application
cors.allowed.origin=http://localhost:5173,http://localhost:5174

# Synthesis settings
# Timeout in seconds for generating the STL file
desyn.stl.timeout=60
# Gurobi solver settings for MIP (Mixed-Integer Programming)
desyn.gurobi.focus=0
desyn.gurobi.gap=0.02
```

---

## ▶️ Running the Application

Once all prerequisites and dependencies are set up, you can run the backend application using the Spring Boot Maven plugin:

```bash
./mvnw spring-boot:run
```
The backend API will be available at `http://localhost:8090`.
Be sure the `.env` file in frontend is configured to point to the correct backend URL (e.g., `http://localhost:8090`). A proxy between the frontend and backend may be required if CORS issues arise.

---

## 📦 Building a JAR File

To build a standalone, executable JAR file for deployment, run the following command:

```bash
./mvnw package
```
The JAR file will be created in the `target/` directory.