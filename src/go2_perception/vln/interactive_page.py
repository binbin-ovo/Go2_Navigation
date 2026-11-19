"""Render the shared camera, native image, trajectory and LiDAR page."""

from pathlib import Path
import re

HERE = Path(__file__).parent
REVIEW_STYLE = (HERE / "review.css").read_text()
TEMPLATE = (HERE / "viewer.html").read_text()


def viewer_page(
    *,
    controls="",
    script="",
    banner="ADVISORY ONLY — NO ROBOT COMMANDS",
    image_loaded="",
    clear_image="",
    grounding="",
    update_controls="panelTest();",
    after_panels="",
    awaiting="No model output yet. Use the Test box above.",
):
    # Only a result still displayed for the current prediction can bind an operator decision.
    if image_loaded:
        image_loaded = (
            "if(!S.prediction||p.prediction_id!==S.prediction.prediction_id)return;"
            + image_loaded
        )
    slots = dict(
        style=REVIEW_STYLE,
        controls=controls,
        script=script,
        banner=banner,
        image_loaded=image_loaded,
        clear_image=clear_image,
        grounding=grounding,
        update_controls=update_controls,
        after_panels=after_panels,
        awaiting=awaiting,
    )
    return re.sub(r"@@([a-z_]+)@@", lambda match: slots[match[1]], TEMPLATE)


PAGE = viewer_page()
