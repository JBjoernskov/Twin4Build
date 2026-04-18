"""F1 -- System diagram (original vs compiled graph).

Builds the synthetic multi-room model at a small ``N_ROOMS`` (3 by default)
and triggers :py:meth:`SimulationModel.visualize` on both the original and
compiled models.  Twin4build writes the graph artefacts under
``generated_files/models/<model_id>/``; this script prints the paths so
they can be copied into the paper's figures folder.

This is a one-shot helper; the other plot scripts are driven from CSVs.
"""

from __future__ import annotations

import sys

from twin4build.examples.paper_experiments import common


N_ROOMS_VIS = 3


def main() -> None:
    print(f"[F1] building {N_ROOMS_VIS}-room model ...")
    model = common.build_multi_room_model(
        N_ROOMS_VIS,
        horizon_days=1,
        step_size=600,
        model_id=f"f1_multi_room_n{N_ROOMS_VIS}",
    )
    model.load(draw_semantic_model=False, draw_simulation_model=True, verbose=0)
    print("[F1]   original: graph written under generated_files/models/")
    print(
        f"[F1]   check: generated_files/models/{model.id}_simulation_model/"
    )

    print("[F1] compiling and drawing ...")
    compiled = model.build_compiled_model()
    compiled.load(
        draw_semantic_model=False, draw_simulation_model=True, verbose=0
    )
    print("[F1]   compiled: graph written under generated_files/models/")
    print(
        f"[F1]   check: generated_files/models/{compiled.id}_simulation_model/"
    )


if __name__ == "__main__":
    sys.exit(main())
