# Flow System Guidelines

*Part of the Flow System. See also: [README](../README.md), [Tasks](../project/tasks.md), [Improvements](../planning/improvements.md).*

This document outlines the core principles and best practices for utilizing the Flow system effectively, ensuring consistency, clarity, and efficient collaboration, especially when working with AI agents.

## Core Principles

1.  **Single Source of Truth**: Each piece of information (task status, configuration, guideline) should reside in a single, designated location.
2.  **Cross-References**: Use markdown links (`[link text](path/to/file.md)`) to connect related information instead of duplicating content.
3.  **Working Memory**: The `flow/` directory, particularly `flow/project/tasks.md`, and the ongoing conversation history serve as the primary working memory. **AI agents MUST actively consult and update this context to maintain awareness of the current state, goals, and history.**
4.  **Clear Communication**: Use concise and unambiguous language in task descriptions, comments, and commit messages.
5.  **Incremental Progress**: Break down large tasks into smaller, manageable steps tracked in `tasks.md`.

## Usage

*   **Tasks (`flow/project/tasks.md`)**: Regularly update task statuses, add new tasks, link related issues/commits, and archive completed items. Use status emojis (e.g., ⏳, ✅, ⏸️, 🔴) for quick visual reference. Ensure task descriptions contain necessary context or links for ongoing work.
*   **Guidelines (`flow/system/guidelines.md`)**: Consult this file for development standards, coding practices, and Flow system usage conventions.
*   **Agent Capabilities (`flow/system/agent_capabilities.md`)**: Refer to this document for the known capabilities and limitations of the assisting AI agent.

## AI Agent Interaction

*   **Context is Key**: Provide clear context when requesting actions. Reference specific files, tasks, or previous messages.
*   **Role Definition**: The AI agent acts as a junior developer. The USER (human developer) provides direction, makes decisions, and reviews work. The AI's role is to execute tasks as instructed, including writing/editing code, running scripts/commands, searching the codebase, and managing Flow documents.
*   **Verify Actions**: Review proposed changes (code edits, commands) before approval.
*   **Iterative Refinement**: Expect to iterate. If the agent's first attempt isn't perfect, provide specific feedback for correction.
*   **Suggesting Next Steps**: When asked by the USER to suggest what to do next, the AI agent MUST consult the current working memory, primarily `flow/project/tasks.md`, to inform its recommendation.
*   **Update Flow**: Ensure the agent updates relevant Flow documents (especially `tasks.md`) after completing actions or changing the project state.

## Overview

The Flow System is designed to manage complex projects by offloading the complexity of task management through a central hub for active tasks. This document serves as the **authoritative source** for all Flow system standards, templates, and guidelines. Other Flow documents should reference these standards rather than duplicating them.

The guidelines outlined here ensure that all active tasks in [tasks.md](../active/tasks.md) have clear pointers to detailed documentation and are kept in sync with broader project information.

## Active Task Management

- **Primary Hub:** Use the [tasks.md](../active/tasks.md) file as the central repository for all active tasks. Each task here should include a brief description and a link to a more detailed document (e.g., a specific project plan or a refactoring breakdown).
- **Task Status:** Include visual indicators for task status (⏳ In progress, ✅ Completed, 🔄 Under review, ⏸️ Paused, 🚩 Blocked).
- **Task Priority:** Mark tasks with priority levels where appropriate (🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low).
- **Status Inference:** A task's activeness is determined by its presence in tasks.md. Explicit status flags make this clearer but are optional.
- **Nested Tasks:** Although tasks may be inherently nested or complex, the system should enforce a single level of active entries in tasks.md. Detailed, granular subtasks should be managed in their respective documentation files to keep the primary list lean.

## System Conventions and Configuration

This section defines standard conventions and configuration preferences used within the Flow system documentation and workflow.

### Appearance

- **Priority Colors**:
  - 🔴 Critical - Used for urgent items that block progress
  - 🟠 High - Important items that need immediate attention
  - 🟡 Medium - Standard work items
  - 🟢 Low - Nice-to-have items or long-term improvements

- **Status Indicators**:
  - ⏳ In progress - Actively being worked on
  - ✅ Completed - Work is finished and verified
  - 🔄 Under review - Awaiting feedback or assessment
  - ⏸️ Paused - Temporarily suspended
  - 🚩 Blocked - Unable to proceed due to dependencies

### Organization

- **Core File Structure** (See `flow/README.md` for full structure):
  - `flow/project/tasks.md` - Primary working document for active project tasks.
  - `flow/system/guidelines.md` - Core documentation and guidelines (this file).
  - `flow/project/conventions.md` - Project-specific conventions.
  - `flow/meta/` - Development info for the Flow system itself.
  - `flow/domains/` - Domain-specific knowledge.

- **Task Categories** (Examples):
  - Implementation - New features or components
  - Bug Fix - Addressing errors or defects
  - Research - Investigation and information gathering
  - Documentation - Improving or extending documentation
  - Refactoring - Restructuring without changing behavior

### Default Behaviors

- **Log Retention** (in `flow/project/logs/`):
  - Keep recent task history in `flow/project/tasks.md`.
  - Archive older completed task entries to the `flow/project/logs/` directory.
  - Retention period for archived logs is project-dependent.

- **Review Frequency** (Recommended):
  - Daily check-ins for active tasks.
  - Weekly review of entire task queue.
  - Periodic consistency audits.

These settings can be adapted based on project requirements, but changes should be documented, ideally in `flow/project/conventions.md` if project-specific.

## Updating Guidelines

- **Frequent Updates:** The tasks.md file should be updated continuously throughout the day. New tasks are added as they arise, and completed or deprioritized tasks are promptly archived.
- **Timestamps & Breadcrumbs:** When initiating a task, record the start time and contextual breadcrumbs to link the task with the appropriate project or subproject.
- **Archival Process:** Once a task is completed, remove it from the active task list and log its completion in the log section with a timestamp.
- **Log Rotation:** Maintain only the most recent logs (5-10 entries) in tasks.md for quick reference. Older logs should be archived to the logs directory with date-based filenames (e.g., logs_2025-03-26.md).
- **Scheduled Reviews & Retrospectives:** Implement regular (e.g., daily/weekly) reviews to reconcile the active task list, ensure consistency, identify blockers, and capture key decisions or outcomes. Consider brief retrospectives after major milestones or debugging sessions to consolidate lessons learned into relevant documentation (e.g., principles, evolution, troubleshooting guides).

## Integration with Other Documents

- **Linking:** Every task in tasks.md must reference the corresponding detailed documentation, such as project plans or other related files.
- **Consistency:** Ensure that updates in detailed task documents are reflected by corresponding updates in the tasks.md file.
- **Context Preservation:** Maintain context between related tasks by:
  - Including clear references to parent/child task relationships
  - Preserving breadcrumbs when tasks branch off from each other
  - Tracking the history of task evolution
  - Using consistent tagging for related tasks (e.g., #refactoring, #debugging)

## Project Boundaries

The Flow system should be treated as a standalone tool that manages the "how" of task tracking, separate from the specific "what" of any given project:

- **Flow Responsibility:** Manages the process, structure, and tracking of tasks
- **Project Responsibility:** Defines the specific content, implementation details, and project-specific files
- **Integration Points:** The tasks.md file serves as the primary integration point between the Flow system and project-specific content

## Task Templates

For common task types, use consistent templates to ensure completeness and clarity:

### Bug Fix Template
```
- **Fix [Bug Name]**: ⏳ [Brief description of the bug]
  - Detailed notes: [Link to detailed bug report]
  - Steps:
    - Identify root cause
    - Implement fix
    - Test solution
    - Update documentation
```

### Feature Implementation Template
```
- **Implement [Feature Name]**: 🟡 ⏳ [Brief feature description]
  - Detailed notes: [Link to feature specification]
  - Dependencies: [Any prerequisite tasks or features]
  - Success criteria: [What defines completion]
```

### Research/Investigation Template
```
- **Investigate [Topic]**: 🟢 ⏳ [Brief description of investigation]
  - Detailed notes: [Link to research notes]
  - Key questions:
    - [Question 1]
    - [Question 2]
  - Timeline: [Expected completion]
```

## Automation Considerations

- **Lightweight Scripting:** Consider using scripts or Git hooks to automate the updating and logging process. Automated prompts or notifications can aid in regular reviews.
- **Log Rotation Automation:** Develop scripts to automatically archive logs older than a defined threshold to the logs directory.
- **Minimize Manual Work:** The primary goal is to reduce the manual overhead of task management by streamlining updates and archival processes.

## Minimizing Cruft and Managing Temporary Files

To prevent the accumulation of unused or temporary files ("cruft"), especially when using automated tools or AI agents for code generation or experimentation, follow these practices:

1.  **Dedicated Scratch/Experiment Directories:**
    *   Utilize a top-level directory named `scratch/` (or similar, e.g., `debug/`) for temporary scripts, outputs, data samples, and exploratory work. Files in these directories are considered ephemeral.

2.  **Permanent Output Location:**
    *   Ensure that *permanent* generated artifacts (model checkpoints, logs, evaluation results, etc.) are placed in the designated project output directory (e.g., `artifacts/` or `outputs/`, check project conventions). Do not leave permanent outputs in `scratch/` or `debug/`.

3.  **Agent Output Containment:**
    *   When collaborating with an AI agent, explicitly instruct it to place all newly generated files within a designated subdirectory (preferably within `scratch/`) unless the files are clearly intended as permanent additions to `src/`, `tests/`, `docs/`, or other core directories.

4.  **Regular Cleanup:**
    *   Periodically review the contents of temporary directories (`scratch/`, `debug/`) and task-specific subdirectories.
    *   Remove files and directories that are no longer needed once an experiment, debugging session, or task is complete. Consider adding a cleanup step to task checklists in `project/active/tasks.md` (or the project-specific task file).
    *   Consult the project's [Troubleshooting and Debugging Guide](../meta/troubleshooting_and_debugging.md#25-cleanup-protocol) for detailed cleanup steps after debugging.

5.  **Explicit Instructions:**
    *   Clearly communicate expectations regarding file placement and lifespan when initiating tasks that involve file creation, especially when working with AI agents.

By adhering to these guidelines, we can maintain a cleaner, more organized, and navigable codebase.

## Maintaining Consistency

This section outlines strategies to prevent inconsistencies in the Flow documentation system, specifically addressing issues related to duplicate information, inconsistent formatting, and lack of clear authoritative sources.

### Key Consistency Principles

1.  **Single Source of Truth**: Each piece of information should be defined in exactly one place (often within this `guidelines.md` file or delegated to project/meta files).
2.  **Cross-References**: Instead of duplicating information, use links (`[link text](path/to/file.md)`) to reference canonical definitions.
3.  **Standardized Formats**: Use consistent formatting (like the status indicators defined above) and naming conventions across all documentation.
4.  **Explicit Ownership**: Files should have clearly defined responsibilities (as outlined in `flow/README.md` and the READMEs within `system/`, `project/`, `meta/`).

### Consistency Maintenance Practices

-   **Regular Audits**:
    -   Periodically audit documentation consistency.
    -   Verify that cross-references remain valid.
    -   Check for duplicated information that should be consolidated.
-   **Change Management**:
    -   When updating standards or guidelines, start with the authoritative source file (often this one).
    -   Update related files as needed.
    -   Document significant changes.
-   **Refactoring Sessions**:
    -   Periodically review the entire Flow documentation system.
    -   Look for opportunities to consolidate or simplify.
    -   Remove outdated information.

### Technical Solutions for Consistency

-   **Automated Validation** (Future Enhancement):
    -   Consider simple scripts to verify cross-references.
    -   Check for consistency in formatting/indicators.
-   **Templates**:
    -   Use standardized templates (like those defined above for tasks) for common documentation elements.
    -   Ensure templates reference authoritative sources rather than duplicating information.

## Conclusion

This document serves as the blueprint for managing task flows effectively. By following these guidelines, the system will remain both flexible and robust, capable of adapting to the complexities of projects while keeping the management process as frictionless as possible. 

For strategies to prevent documentation inconsistencies and maintain the integrity of the Flow system, refer to the [Consistency](consistency.md).

## AI Agent Interaction with Flow

To enable AI agents (like coding assistants) to effectively utilize the Flow system as a form of working memory and context provider with minimal explicit prompting, agents should adhere to the following information retrieval protocol:

1.  **Parse Root README:** Always begin by processing `flow/README.md`. This provides the high-level structure, purpose of main directories, and pointers to other key files.

2.  **Check Active Tasks:** Consult `flow/project/active/tasks.md` to understand the immediate priorities and current work context.

3.  **Consult Subdirectory READMEs:** Based on the nature of the current task (e.g., adhering to standards, understanding project plans, needing meta-context), parse the relevant subdirectory `README.md` (`system/README.md`, `project/README.md`, `meta/README.md`) for more specific pointers within that section.

4.  **Utilize Semantic Search:** If the required information isn't immediately located via READMEs or direct links, use semantic search queries targeted within the `flow/` directory. Keywords from the task description and relevant READMEs should inform the search query.

5.  **Read Specific Files Contextually:** Only read the full contents of specific guideline files (like this one), principle documents, planning documents, or reference files if:
    *   They are directly linked from a task or README.
    *   They are identified as highly relevant by a semantic search.
    *   The current task explicitly requires understanding or adhering to a standard defined within that specific file.

This prioritized retrieval process aims to provide the necessary context efficiently, leveraging the structured information within Flow without requiring the agent to read every file unnecessarily.

## Adherence to Conventions

Strive to follow common community conventions for Python project structure, coding style (e.g., PEP 8), naming, documentation, and tooling unless there is a documented and compelling reason specific to this project to deviate. This promotes consistency, readability, and interoperability. [TODO: This should be generalized, with Python-specifics moved to a Python domain file.]