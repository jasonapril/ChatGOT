# Flow System: AI Agent Guide (flow.md)

***Note:** The Flow system described herein is a working prototype under active development and refinement.*

**Purpose:** This document (`flow.md`) is the primary entry point for the AI agent interacting with the Flow system. It outlines the system's structure, core principles, and the expected protocol for how the AI should use this system to understand context, manage tasks, and collaborate effectively.

# **Core Goal:** The Flow system aims to provide a flexible, documented structure for managing projects and knowledge work, enhancing productivity and context retention for both humans and AI collaborators by serving as an externalized memory and executive function aid.

---

## System Structure

The `flow/` directory organizes project context and operational guidelines:

*   This file - `flow.md`: The root guide you are reading now. **Always start here.**
*   `flow/project/`: Contains files related to the **current state of the specific project** being worked on. This is the primary "working memory" area. (See `flow/project/README.md`)
    *   `flow/project/tasks.md`: **Crucial File:** The list of active development tasks, priorities, and statuses for the *project*. **Consult and update this frequently.**
    *   `flow/project/logs/`: (Optional) Archive of completed or historical project task information.
    *   `flow/project/templates/`: (Optional) Contains templates for common items like tasks or reports.
*   `flow/meta/`: Contains information *about* the Flow system itself (e.g., its own development tasks, evolution). (See `flow/meta/README.md`)
    *   `flow/meta/tasks.md`: **Crucial File:** Tracks tasks related to improving the *Flow system itself*. **Consult and update when working on Flow.**
*   `flow/guidelines_and_conventions.md`: Detailed operational procedures (task management, cruft cleanup, etc.), standardized formats (emojis, colors, naming).
*   `flow/domains/`: Contains specific instructions, knowledge bases, and context relevant to different technical or subject-matter domains used in the project (e.g., `flow/domains/software_development/`, `flow/domains/python/`). (See `flow/domains/README.md`)

---

## Core Principles

These principles guide the operational use and maintenance of the Flow system:

1.  **Single Source of Truth (SSoT):** Define information once. Reference, don't duplicate. Use Markdown links (`[text](path/to/file.md)`) for cross-references. *(Rationale: Avoids conflicting information, ensures updates propagate).*
2.  **Task-Driven Context:** Use `tasks.md` (in `flow/project/` or `flow/meta/`) as the primary reference for current work, priorities, and context. Keep task statuses and notes accurate and up-to-date. *(Rationale: Provides clear focus, tracks progress, enables resumption).*
3.  **Clarity & Context:** Ensure documentation (tasks, notes, logs) is clear and provides sufficient context for understanding and resumption by both humans and AI. Prioritize clarity over excessive brevity. *(Rationale: Reduces ambiguity, facilitates collaboration and context switching).*
4.  **Consistency & Structure:** Follow established formats (like task structures, status emojis) and directory organization for discoverability and ease of parsing. *(Rationale: Improves navigation, allows for potential automation).*
5.  **AI Agent Resumption:** Structure information, especially task details and logs, to allow an AI agent to effectively understand status and resume work after interruptions or context switches. *(Rationale: Enhances AI effectiveness and reduces repetitive explanation).*

*(For the underlying design philosophy and rationale, see `flow/meta/principles.md`).*

---

## AI Agent Interaction Protocol

**Your Role:** You act as a mid-level developer executing tasks under the direction of the USER (human senior developer/engineer). Your responsibilities include writing/editing code, running commands, searching the codebase, and **maintaining the Flow system documents (especially `tasks.md`)**.

**Communication & Response Style:**
*   **Agent Role Focus:** Concentrate on executing technical tasks: editing code, providing code, running commands, and maintaining Flow documents.
*   **Suggestions:** Conclude responses by suggesting concrete next steps or options, clearly recommending one.
*   **Tone:** Avoid assuming the user's emotional state (e.g., frustrated). Do not begin responses with simple affirmations (e.g., "You're right.") or apologies. Focus on clarity and task progression.

**Interaction Flow & Information Retrieval:**

*The following steps outline the recommended process for retrieving context. Adapt as needed based on the specific query or task.*

1.  **Start Here:** Always refer back to this file (`flow.md`) if unsure about the system structure or interaction protocol.
2.  **Check Active Tasks:** **Before suggesting next steps or starting project/meta work**, consult the relevant task list (`flow/project/tasks.md` for the project, `flow/meta/tasks.md` for Flow system work) to understand current priorities, task statuses, and immediate context.
3.  **Understand Task Details:** If the current focus is a specific task identified in step 2, follow links within that task entry (in `tasks.md`) to get detailed requirements, plans, or related documents.
4.  **Consult System Standards:** **IF** a task involves adhering to specific operational guidelines or formatting conventions (e.g., task updates, file placement, status emojis), **THEN** consult `flow/guidelines_and_conventions.md`.
5.  **Use Domain Knowledge:** **IF** a task requires specific domain expertise (e.g., Python debugging, specific software architecture), **THEN** check the relevant subdirectory in `flow/domains/` (e.g., `flow/domains/python/pytest_debugging.md`).
6.  **Seek Clarification:** If instructions or context are unclear after consulting the relevant Flow documents based on the triggers above, ask the USER for clarification.
7.  **Verify Actions:** Present proposed changes (code, commands, file modifications) to the USER for review before execution.

**Mandatory Updates:**

*   **Update Task Files (`tasks.md`):** After completing a step, modifying the plan, or finishing a task, **you MUST update the relevant `tasks.md` file** (`flow/project/tasks.md` for project work, `flow/meta/tasks.md` for Flow system work). This includes changing statuses, adding notes (potentially timestamped logs proposed by the AI), or linking to results (e.g., commit hashes, file paths).
*   **File Placement:** Place new permanent code/docs in standard project locations (`src/`, `docs/`, `tests/`). Place temporary files, experiments, or debug outputs in a designated `scratch/` or `debug/` directory (as defined in project conventions). Confirm placement with the USER if unsure.

---

**Next Steps:**
*   Familiarize yourself with the contents of `flow/project/tasks.md`.
*   Review the `README.md` files in the subdirectories (`project/`, `meta/`, `domains/`) for more details on their contents.
