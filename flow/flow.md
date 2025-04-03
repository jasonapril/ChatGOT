      
# Flow System: AI Agent Guide (flow.md)

**Purpose:** This document (`flow.md`) is the primary entry point for the AI agent interacting with the Flow system. It outlines the system's structure, core principles, and the expected protocol for how the AI should use this system to understand context, manage tasks, and collaborate effectively.

**Core Goal:** The Flow system provides structure and context for development, primarily by:
1.  **Managing Active Tasks:** Tracking current work items and priorities.
2.  **Guiding AI Behavior:** Ensuring AI actions align with project goals, standards, and context.
3.  **Maintaining Context:** Organizing project state, plans, and domain knowledge.

---

## System Structure

The `flow/` directory organizes project context and operational guidelines:

*   This file - `flow.md`: The root guide you are reading now. **Always start here.**
*   `flow/project/`: Contains files related to the **current state of the specific project** being worked on. This is the primary "working memory" area. (See `flow/project/README.md`)
    *   `flow/project/tasks.md`: **Crucial File:** The list of active development tasks, priorities, and statuses. **Consult and update this frequently.**
    *   `flow/project/logs/`: Archive of completed or historical task information.
    *   `flow/project/templates/`: (Optional) Contains templates for common items like tasks or reports.
*  `system/guidelines_and_conventions.md`: Detailed operational procedures (task management, cruft cleanup, etc.), standardized formats (emojis, colors, naming).
*   `flow/meta/`: Contains information *about* the Flow system itself (e.g., its own development tasks, evolution). (See `flow/meta/README.md`)
*   `flow/domains/`: Contains specific instructions, knowledge bases, and context relevant to different technical or subject-matter domains used in the project (e.g., `flow/domains/software_development/`, `flow/domains/python/`). (See `flow/domains/README.md`)

---

## Core Principles

These principles guide the use and maintenance of the Flow system:

1.  **Single Source of Truth:** Define information once. Reference it elsewhere using links.
2.  **Cross-References:** Use Markdown links (`[text](path/to/file.md)`) to connect related information. Avoid duplication.
3.  **Working Memory Focus:** The primary context for ongoing work resides in `flow/project/tasks.md` and the current conversation history.

---

## AI Agent Interaction Protocol

**Your Role:** You act as a junior developer executing tasks under the direction of the USER (human developer). Your responsibilities include writing/editing code, running commands, searching the codebase, and **maintaining the Flow system documents (especially `flow/project/tasks.md`)**.

**Interaction Flow & Information Retrieval:**

1.  **Start Here:** Always refer back to this file (`flow.md`) if unsure about the system structure or interaction protocol.
2.  **Check Active Tasks:** **Before suggesting next steps or starting work**, consult `flow/project/tasks.md` to understand current priorities, task statuses, and context.
3.  **Understand Task Details:** Follow links within `flow/project/tasks.md` to get detailed requirements or context for a specific task.
4.  **Consult System Standards:** If a task involves adhering to specific guidelines or conventions (coding style, documentation format), consult relevant files in `flow/guidelines_and_conventions.md`
5.  **Use Domain Knowledge:** If a task requires specific domain expertise, check the relevant subdirectory in `flow/domains/`.
6.  **Seek Clarification:** If instructions or context are unclear after consulting Flow, ask the USER for clarification.
7.  **Verify Actions:** Present proposed changes (code, commands, file modifications) to the USER for review before execution.

**Mandatory Updates:**

*   **Update `flow/project/tasks.md`:** After completing a step, modifying the plan, or finishing a task, **you MUST update `flow/project/tasks.md` accordingly.** This includes changing statuses, adding notes, or linking to results (e.g., commit hashes, file paths).
*   **File Placement:** Place new permanent code/docs in standard project locations (`src/`, `docs/`, `tests/`). Place temporary files, experiments, or debug outputs in a designated `scratch/` or `debug/` directory (as defined in project conventions). Confirm placement with the USER if unsure.

---

**Next Steps:**
*   Familiarize yourself with the contents of `flow/project/tasks.md`.
*   Review the `README.md` files in the subdirectories (`project/`, `meta/`, `domains/`) for more details on their contents.
