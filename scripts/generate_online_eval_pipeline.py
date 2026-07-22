#!/usr/bin/env python3
"""Generate the branch-clone Harness online eval pipeline YAML."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".harness" / "online_eval.yaml"


def _current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _pipeline_yaml(branch: str) -> str:
    branch_default = branch or "feat/harness-otel-eval-case-source"
    return f"""pipeline:
  name: Online Evals Trace TaskCompletion
  identifier: online_evals_trace_task_completion
  orgIdentifier: SrikarOrg
  projectIdentifier: SrikarProject
  tags:
    purpose: online-evals
  properties:
    ci:
      codebase:
        connectorRef: <+pipeline.variables.repo_connector>
        repoName: <+pipeline.variables.repo_name>
        build:
          type: branch
          spec:
            branch: <+pipeline.variables.branch>
  variables:
    - name: trace_ids
      type: String
      description: Comma-separated trace IDs to evaluate.
      required: true
      value: <+input>
    - name: org_id
      type: String
      description: Org whose traces to query with Harness QueryService.
      required: true
      value: <+input>.default(SrikarOrg)
    - name: project_id
      type: String
      description: Project whose traces to query with Harness QueryService.
      required: true
      value: <+input>.default(SrikarProject)
    - name: branch
      type: String
      description: Branch ref to clone. Use the current feature branch until these changes are published.
      required: true
      value: <+input>.default({branch_default})
    - name: repo_connector
      type: String
      description: Connector used to clone this repository.
      required: true
      value: <+input>
    - name: repo_name
      type: String
      description: Repository name/path for the connector.
      required: true
      value: <+input>.default(harness-evals)
    - name: docker_connector
      type: String
      description: Connector used to pull the Python image.
      required: true
      value: <+input>.default(dockerHubAnon)
    - name: harness_api_key
      type: Secret
      description: Harness PAT or SAT with QueryService access.
      required: true
      value: <+input>
    - name: openai_api_key
      type: Secret
      description: OpenAI API key for TaskCompletion judge.
      required: true
      value: <+input>
    - name: harness_base_url
      type: String
      description: Harness base URL.
      required: false
      value: <+input>.default(https://app.harness.io)
    - name: threshold
      type: String
      description: TaskCompletion pass threshold.
      required: false
      value: <+input>.default(0.7)
    - name: model
      type: String
      description: OpenAI judge model.
      required: false
      value: <+input>.default(gpt-4o-mini)
  stages:
    - stage:
        name: Online Eval
        identifier: online_eval
        type: CI
        spec:
          cloneCodebase: true
          platform:
            os: Linux
            arch: Amd64
          runtime:
            type: Cloud
            spec: {{}}
          execution:
            steps:
              - step:
                  type: Run
                  name: Fetch and Score Traces
                  identifier: fetch_and_score
                  timeout: 45m
                  spec:
                    connectorRef: <+pipeline.variables.docker_connector>
                    image: python:3.11-slim
                    shell: Bash
                    command: |-
                      set -euo pipefail

                      echo "=== Install branch-local harness-evals ==="
                      python -m pip install --upgrade pip
                      python -m pip install -e ".[harness,llm]"

                      echo "=== Run online eval ==="
                      mkdir -p /harness/results
                      python scripts/online_eval.py \\
                        --trace-ids "<+pipeline.variables.trace_ids>" \\
                        --org-id "<+pipeline.variables.org_id>" \\
                        --project-id "<+pipeline.variables.project_id>" \\
                        --threshold "<+pipeline.variables.threshold>" \\
                        --model "<+pipeline.variables.model>" \\
                        --output-dir /harness/results

                      echo "=== Results ==="
                      ls -la /harness/results
                      python -m json.tool /harness/results/scores.json
                    envVariables:
                      HARNESS_API_KEY: <+pipeline.variables.harness_api_key>
                      HARNESS_ACCOUNT_ID: <+account.identifier>
                      HARNESS_BASE_URL: <+pipeline.variables.harness_base_url>
                      OPENAI_API_KEY: <+pipeline.variables.openai_api_key>
                    reports:
                      type: JUnit
                      spec:
                        paths:
                          - /harness/results/junit.xml
        failureStrategies:
          - onFailure:
              errors:
                - AllErrors
              action:
                type: MarkAsFailure
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(_pipeline_yaml(_current_branch()), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
