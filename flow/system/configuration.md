# Flow System Configuration

*Part of the Flow System. See also: [Guidelines](guidelines.md), [Tasks](../active/tasks.md).*

## Overview

This document defines the configuration preferences for the Flow system, establishing consistent settings and behaviors across all documentation. These configurations ensure a uniform workflow while allowing for customization based on project needs.

## Configuration Options

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

- **File Structure**:
  - `active/tasks.md` - Primary working document
  - `system/` - Core documentation and guidelines
  - `planning/` - Future-oriented documents and improvements
  - `reference/` - Implementation details and specialized information
  - `logs/` - Historical records and archives

- **Task Categories**:
  - Implementation - New features or components
  - Bug Fix - Addressing errors or defects
  - Research - Investigation and information gathering
  - Documentation - Improving or extending documentation
  - Refactoring - Restructuring without changing behavior

### Default Behaviors

- **Log Retention**:
  - Keep 5-10 most recent entries in tasks.md
  - Archive older entries to logs directory
  - Retain archived logs for at least 6 months

- **Review Frequency**:
  - Daily check-ins for active tasks
  - Weekly review of entire task queue
  - Bi-weekly consistency audits
  - Monthly system evaluation

## Custom Settings

These settings can be modified based on project requirements. When changing a setting, update this document first, then apply the changes consistently across other documents.

## Implementation Details

For technical implementation details of configuration loading and application, see [Config Implementation](../reference/config_implementation.md). 