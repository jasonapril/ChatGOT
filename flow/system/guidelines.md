# Flow System Guidelines

*Part of the Flow System. See also: [Tasks](../active/tasks.md), [Consistency](consistency.md).*

## Overview

The Flow System is designed to manage complex projects by offloading the complexity of task management through a central hub for active tasks. This document serves as the **authoritative source** for all Flow system standards, templates, and guidelines. Other Flow documents should reference these standards rather than duplicating them.

The guidelines outlined here ensure that all active tasks in [tasks.md](../active/tasks.md) have clear pointers to detailed documentation and are kept in sync with broader project information.

## Active Task Management

- **Primary Hub:** Use the [tasks.md](../active/tasks.md) file as the central repository for all active tasks. Each task here should include a brief description and a link to a more detailed document (e.g., a specific project plan or a refactoring breakdown).
- **Task Status:** Include visual indicators for task status (⏳ In progress, ✅ Completed, 🔄 Under review, ⏸️ Paused, 🚩 Blocked).
- **Task Priority:** Mark tasks with priority levels where appropriate (🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low).
- **Status Inference:** A task's activeness is determined by its presence in tasks.md. Explicit status flags make this clearer but are optional.
- **Nested Tasks:** Although tasks may be inherently nested or complex, the system should enforce a single level of active entries in tasks.md. Detailed, granular subtasks should be managed in their respective documentation files to keep the primary list lean.

## Updating Guidelines

- **Frequent Updates:** The tasks.md file should be updated continuously throughout the day. New tasks are added as they arise, and completed or deprioritized tasks are promptly archived.
- **Timestamps & Breadcrumbs:** When initiating a task, record the start time and contextual breadcrumbs to link the task with the appropriate project or subproject.
- **Archival Process:** Once a task is completed, remove it from the active task list and log its completion in the log section with a timestamp.
- **Log Rotation:** Maintain only the most recent logs (5-10 entries) in tasks.md for quick reference. Older logs should be archived to the logs directory with date-based filenames (e.g., logs_2025-03-26.md).
- **Scheduled Reviews:** Implement regular (e.g., nightly) reviews to reconcile the active task list, ensuring consistency and identifying any tasks that may require further attention.

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

## Conclusion

This document serves as the blueprint for managing task flows effectively. By following these guidelines, the system will remain both flexible and robust, capable of adapting to the complexities of projects while keeping the management process as frictionless as possible. 

For strategies to prevent documentation inconsistencies and maintain the integrity of the Flow system, refer to the [Consistency](consistency.md). 