# Flow System

This directory contains the Flow system, designed to manage development workflow, guide AI agent behavior, and maintain project context.

## Structure

The Flow system is organized into the following main subdirectories:

*   `system/`: Contains core principles, development guidelines, coding standards, and consistency rules. These define the foundational practices for the project.
*   `project/`: Contains files related to the state of the project.
*   `meta/`: Contains development project for the Flow system itself.
*   `domains/`: Contains instructions and domain knowledge for the AI agent. Different projects will utilize different domains.

Consult the `README.md` file within each subdirectory (if present) for more details on their specific contents.

## Purpose

The primary goals of the Flow system are:

1.  **Task Management:** Provide a clear view of active tasks, priorities, and progress.
2.  **Consistency:** Enforce standards and guidelines (defined in `system/`).
3.  **AI Guidance:** Offer a structured context for AI agents involved in development, ensuring their actions align with project goals and standards.
4.  **Context Preservation:** Keep project-specific information (tasks, plans, references) organized and accessible within the `project/` subdirectory.

## Usage

- Regularly update `project/tasks.md` to reflect the current work.
- Consult `system/` for development guidelines and standards.
  - See [system/guidelines.md](system/guidelines.md) for a complete overview of how the Flow system works and the best practices for using it effectively.

## Core Principles

- **Single Source of Truth**: Each piece of information is defined in one place
- **Cross-References**: Use links instead of duplicating information
- **Working Memory**: The active directory serves as a centralized hub