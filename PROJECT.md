# Product Requirements Document: JAX TPU Lab

## 1. Overview

### 1.1. Vision

To create a seamless, web-based Integrated Development Environment (IDE) that empowers developers and researchers to write, review, and execute JAX code on Google's Tensor Processing Units (TPUs) efficiently and securely.

### 1.2. Problem Statement

Developing and optimizing JAX code for TPUs often requires a complex local setup, access to specialized hardware, and a disjointed workflow for code review and performance analysis. This creates a high barrier to entry and slows down the iteration cycle.

### 1.3. Goal

The JAX TPU Lab will provide a unified, cloud-native platform that integrates a code editor, AI-powered code review, and a secure execution environment, allowing users to go from idea to optimized TPU code with minimal friction.

## 2. User Personas

*   **ML Researcher/Engineer:** Needs to quickly prototype and test JAX models on TPUs. Cares about performance, optimization, and rapid iteration.
*   **Student/Hobbyist:** Wants to learn JAX and experiment with TPUs without the overhead of managing infrastructure.

## 3. Functional Requirements

The application is composed of three main pillars: a web-based frontend, a serverless backend for control and state management, and a secure compute environment for code execution.

### 3.1. Frontend (React Application)

The user interface must be a clean, modern, and responsive single-page application.

*   **Header:**
    *   **Application Title:** Displays "JAX TPU Lab".
    *   **Authentication:** Provides a "Sign In" button that, upon successful authentication, displays a welcome message with the user's email (e.g., "Welcome, user@email.com").

*   **Main Section (Editor & Tabs):**
    *   **Code Editor:** Must use the Monaco Editor for a rich code editing experience.
    *   **Tabbed Interface:**
        *   `main.py`: A read/write tab for the user's primary JAX code.
        *   `README.md`: A read-only tab to display the problem description or instructions (supports Markdown).
        *   `requirements.txt`: A read/write tab for specifying Python package dependencies.
    *   **Action Buttons:**
        *   **"Review Code":** Triggers an AI-powered code review workflow.
        *   **"Run on TPU":** Triggers the asynchronous job execution workflow.

*   **Bottom Panel (Results & Logs):**
    *   **Real-time Updates:** The panel must listen for real-time updates from the active job's document in Firestore.
    *   **Job Status:** Displays the current status of the job (e.g., `QUEUED`, `COMPILING`, `RUNNING`, `SUCCESS`, `FAILED`).
    *   **Logs:** Streams and displays `stdout` and `stderr` from the job execution in real-time.
    *   **Performance Metrics:** Displays final metrics (e.g., execution time, compilation time, accuracy) once the job is complete.

### 3.2. Backend (Firebase)

*   **Authentication:**
    *   Must use **Firebase Authentication** with **Google Sign-In** as the primary provider.
    *   All user-generated data (code, jobs) must be scoped to the authenticated user's ID (`userId`).

*   **Database (Firestore):**
    *   Firestore will be the source of truth for user and job data.
    *   **Security Rules:** Strict security rules must be implemented to ensure users can only access their own data.
    *   **Data Structure:**
        *   `users/{userId}/code/{docId}`: Stores user's code files (`main.py`, `requirements.txt`).
        *   `users/{userId}/jobs/{jobId}`: Stores the state for each execution job.
            *   `status`: (string) `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`.
            *   `code`: (string) A snapshot of `main.py` at the time of execution.
            *   `logs`: (string) Real-time log output.
            *   `metrics`: (map) e.g., `{ compilationTime, executionTime, accuracy }`.
            *   `createdAt`, `finishedAt`: (timestamp).

*   **Control Plane API (Firebase Functions):**
    *   **`reviewCode(code: string)`:**
        *   An HTTP-triggered function that accepts user code.
        *   Constructs a system prompt for the Gemini API, instructing it to act as an expert in JAX and TPU optimization.
        *   Calls the Gemini API to generate a code review.
        *   Returns the AI-generated feedback as a Markdown string.
    *   **`runJob(code: string, requirements: string)`:**
        *   An HTTP-triggered function that initiates an asynchronous job.
        *   Retrieves the authenticated `userId`.
        *   Creates a new job document in Firestore under `users/{userId}/jobs/{jobId}` with an initial status of `QUEUED`.
        *   **Triggers the secure execution harness on the TPU server**, passing the `jobId` and `userId`.
        *   Immediately returns the `jobId` to the frontend.

### 3.3. Compute (Secure TPU Execution Environment)

*   **Execution Model:**
    *   User code will **not** be executed via a general-purpose container orchestration system like GKE.
    *   Instead, execution will occur on a dedicated server co-located with the TPU hardware.

*   **Hardened Harness:**
    *   A secure execution harness is responsible for running user code.
    *   The harness **must not** execute the user's script in its entirety. It will only invoke specific, designated functions within the script (e.g., a `main()` or `train_step()` function). This is a critical security and sandboxing requirement.

*   **Runner Script (`runner.py`):**
    *   This script acts as the entrypoint for the execution harness.
    *   **Initialization:**
        *   Reads `JOB_ID` and `USER_ID` from environment variables.
        *   Initializes the `firebase-admin` SDK to communicate with Firestore.
        *   Retrieves the job document reference: `db.collection('users').doc(USER_ID).collection('jobs').doc(JOB_ID)`.
    *   **Execution and Logging:**
        *   Fetches the user's code (`main.py`) and dependencies (`requirements.txt`) from the job document.
        *   Updates the job status in Firestore to `COMPILING`.
        *   Installs the specified dependencies in a sandboxed environment.
        *   Executes the designated function from the user's script as a subprocess.
        *   Updates the job status to `RUNNING`.
        *   **Crucially, it must capture `stdout` and `stderr` from the subprocess in real-time and append them to the `logs` field in the Firestore job document.** This enables live log streaming to the UI.
    *   **Completion:**
        *   Upon subprocess completion, captures the exit code.
        *   Parses the logs for key performance metrics (e.g., lines matching "Final accuracy: 0.92").
        *   Performs a final update to the job document with:
            *   `status`: `COMPLETED` or `FAILED`.
            *   `metrics`: A map of parsed performance data.
            *   `finishedAt`: A server timestamp.

## 4. Technical Stack

*   **Frontend:** React, Tailwind CSS (delivered as a single `.jsx` file).
*   **Backend & Database:** Firebase (Authentication, Firestore, Functions).
*   **Compute:** A dedicated server with access to a TPU, running a secure Python execution harness.

## 5. Non-Functional Requirements

*   **Security:** User code must be executed in a sandboxed environment. The execution harness must prevent access to the underlying system and only permit the execution of specific functions. All database access must be protected by strict Firestore security rules.
*   **Scalability:** The control plane (Firebase) is serverless and scales automatically. The compute environment must be designed to handle multiple concurrent jobs.
*   **Usability:** The interface should be intuitive and provide immediate feedback, especially through real-time logging and status updates.
