# Flow Project Directory

## Vision / Goals

*(See [vision_and_plan.md](vision_and_plan.md) for detailed project goals and the development plan.)*

## Overview

This directory contains files specific to the current project's state, tasks,
and conventions, supplementing the core Flow system.

## Relevant Domains

This project utilizes domain knowledge from the following areas:

- **Software Development**: General principles, guidelines, and troubleshooting for software engineering. See [../domains/software_development/README.md](../domains/software_development/README.md).

# Project Directory

This directory holds all information specific to the project currently being managed by the Flow system.

## Purpose

The `project/` directory separates the dynamic, project-specific context (tasks, plans, logs, references) from the stable, system-level guidelines (`system/`) and the meta-documentation about Flow itself (`meta/`).

## Structure

*   **`tasks.md`**: Tracks active development tasks, priorities, and status.
*   **`vision_and_plan.md`**: Outlines the project vision, high-level goals, and the phased development plan.
*   **`conventions.md`**: Documents project-specific conventions (e.g., directory structures, naming).
*   **`logs/`**: Archives historical records, such as completed task logs or event timelines.
*   **`README.md`**: This file.

### Working Memory

- **`tasks.md`** reflects *what is happening now*.

### Long-term Memory

- **`reference.md`**: Stores project-specific technical documentation, configuration details, or other reference materials needed for the current project.
- **`logs/`**: Archives historical records, such as completed task logs or event timelines.

### Executive Hierarchy

Documents related to future work, such as roadmaps, specific feature plans, or improvement proposals.

- **`vision_and_plan.md`**

# Project Context (`flow/project/`)

This directory holds the active context for the specific project being worked on (e.g., Craft).

## Contents

*   `README.md`: This overview file.
*   `tasks.md`: The primary list of active project tasks, priorities, and statuses.
*   `logs/`: Directory containing archived logs of completed project tasks.
*   `templates/`: (Optional) Contains templates for project-specific items.
*   `reference_minimal_test_run_gpu.md`: Notes documenting a known-good baseline configuration and environment for running a minimal training test on a specific GPU (GTX 1650 Ti), useful for debugging environment or basic configuration issues.

*(Project-specific configuration files, detailed plans, or temporary notes may also reside here or be linked from `tasks.md`)*