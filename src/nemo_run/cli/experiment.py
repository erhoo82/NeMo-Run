# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Annotated
import typer

from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.run.experiment import Experiment


def _get_experiment(experiment_id: str) -> Experiment:
    try:
        exp = Experiment.from_id(experiment_id)
    except Exception:
        exp = Experiment.from_title(experiment_id)

    assert exp, f"Experiment {experiment_id} not found."
    return exp


def list(experiment_title: str):
    """List all experiments for a given title."""
    CONSOLE.log(f"[bold magenta] Listing experiments with title {experiment_title}")
    CONSOLE.log(Experiment.catalog(experiment_title))


def logs(experiment_id: str, task_idx: Annotated[int, typer.Argument()] = 0):
    """
    Show logs for an experiment task for the experiment id/title
    and optional task_idx (0 by default).
    """

    exp = _get_experiment(experiment_id)

    with exp:
        exp.logs(task_id=exp.tasks[task_idx].id)


def status(experiment_id: str):
    """
    Show status for an experiment given its id/title.
    """
    exp = _get_experiment(experiment_id)

    with exp:
        exp.status()


def cancel(experiment_id: str, task_idx: Annotated[int, typer.Argument()] = 0):
    """
    Cancel an experiment task for the experiment id/title
    and optional task_idx (0 by default).
    """
    exp = _get_experiment(experiment_id)

    with exp:
        exp.cancel(task_id=exp.tasks[task_idx].id)


def create() -> typer.Typer:
    app = typer.Typer()

    app.command(
        "logs",
        context_settings={"allow_extra_args": False},
    )(logs)
    app.command(
        "list",
        context_settings={"allow_extra_args": False},
    )(list)
    app.command(
        "status",
        context_settings={"allow_extra_args": False},
    )(status)
    app.command(
        "cancel",
        context_settings={"allow_extra_args": False},
    )(cancel)

    return app