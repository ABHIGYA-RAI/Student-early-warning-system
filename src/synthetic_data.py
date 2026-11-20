
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import ensure_dir, set_seed


def _sample_choice(rng: np.random.Generator, choices, probs=None, size=None):
    return rng.choice(choices, size=size, replace=True, p=probs)


def generate_sample_oulad(output_dir: str | Path, seed: int = 42, n_students: int = 480) -> None:
    """Generate a small OULAD-like synthetic dataset for smoke testing."""
    output_dir = ensure_dir(output_dir)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    modules = ["AAA", "BBB", "CCC", "DDD"]
    presentations = ["2013B", "2013J", "2014B", "2014J"]

    courses = []
    for m in modules:
        for p in presentations:
            courses.append({"code_module": m, "code_presentation": p, "length": int(rng.integers(200, 270))})
    courses_df = pd.DataFrame(courses)

    assessment_rows = []
    id_assessment = 1
    for row in courses:
        for week in [2, 4, 6, 8]:
            assessment_rows.append({
                "code_module": row["code_module"],
                "code_presentation": row["code_presentation"],
                "id_assessment": id_assessment,
                "assessment_type": "TMA" if week < 8 else "CMA",
                "date": week * 7,
                "weight": 25
            })
            id_assessment += 1
    assessments_df = pd.DataFrame(assessment_rows)

    vle_rows = []
    site_id = 1
    for row in courses:
        for activity_type in ["forumng", "oucontent", "resource", "subpage", "quiz"]:
            vle_rows.append({
                "id_site": site_id,
                "code_module": row["code_module"],
                "code_presentation": row["code_presentation"],
                "activity_type": activity_type,
                "week_from": int(rng.integers(1, 6)),
                "week_to": int(rng.integers(6, 12))
            })
            site_id += 1
    vle_df = pd.DataFrame(vle_rows)

    student_info = []
    student_reg = []
    student_assessment = []
    student_vle = []

    vle_lookup = vle_df.groupby(["code_module", "code_presentation"])["id_site"].apply(list).to_dict()
    assess_lookup = assessments_df.groupby(["code_module", "code_presentation"])

    for sid in range(100000, 100000 + n_students):
        mod = str(rng.choice(modules))
        pres = str(rng.choice(presentations))
        gender = str(_sample_choice(rng, ["M", "F"], probs=[0.48, 0.52]))
        disability = str(_sample_choice(rng, ["N", "Y"], probs=[0.9, 0.1]))
        age_band = str(_sample_choice(rng, ["0-35", "35-55", "55<="], probs=[0.70, 0.24, 0.06]))
        highest_education = str(_sample_choice(
            rng,
            ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"],
            probs=[0.05, 0.18, 0.42, 0.26, 0.09]
        ))
        imd_band = str(_sample_choice(
            rng,
            ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
        ))
        prev_attempts = int(rng.integers(0, 4))
        credits = int(_sample_choice(rng, [30, 60, 90, 120], probs=[0.10, 0.55, 0.23, 0.12]))
        date_registration = int(rng.integers(-40, 10))

        risk = 0.15
        risk += 0.10 if disability == "Y" else 0.0
        risk += 0.06 if age_band == "55<=" else 0.0
        risk += 0.04 if highest_education in ["No Formal quals", "Lower Than A Level"] else -0.01
        risk += 0.05 if credits >= 90 else 0.0
        risk += 0.04 if prev_attempts >= 2 else 0.0
        risk += 0.02 if gender == "M" else 0.0
        risk = float(np.clip(risk + rng.normal(0, 0.03), 0.03, 0.75))

        withdrew = rng.random() < risk
        final_result = "Withdrawn" if withdrew else str(_sample_choice(rng, ["Pass", "Fail", "Distinction"], probs=[0.68, 0.18, 0.14]))
        date_unregistration = int(rng.integers(20, 120)) if withdrew else np.nan

        student_info.append({
            "code_module": mod,
            "code_presentation": pres,
            "id_student": sid,
            "gender": gender,
            "region": "Region_" + str(rng.integers(1, 8)),
            "highest_education": highest_education,
            "imd_band": imd_band,
            "age_band": age_band,
            "num_of_prev_attempts": prev_attempts,
            "studied_credits": credits,
            "disability": disability,
            "final_result": final_result
        })

        student_reg.append({
            "code_module": mod,
            "code_presentation": pres,
            "id_student": sid,
            "date_registration": date_registration,
            "date_unregistration": date_unregistration
        })

        assess_df = assess_lookup.get_group((mod, pres)).copy()
        base_score = 65 - 18 * risk + rng.normal(0, 8)
        for _, arow in assess_df.iterrows():
            submitted = (not withdrew) or (arow["date"] < date_unregistration)
            if submitted:
                delay = int(rng.integers(-1, 5))
                score = float(np.clip(base_score + rng.normal(0, 12), 5, 100))
                student_assessment.append({
                    "id_assessment": int(arow["id_assessment"]),
                    "id_student": sid,
                    "date_submitted": int(arow["date"] + delay),
                    "is_banked": 0,
                    "score": round(score, 2)
                })

        sites = vle_lookup[(mod, pres)]
        engagement = 1.0 - risk + rng.normal(0, 0.08)
        engagement = float(np.clip(engagement, 0.1, 1.6))
        max_day = int(date_unregistration) if withdrew else 70
        for day in range(0, max_day, 7):
            active_sites = rng.choice(sites, size=int(rng.integers(1, min(4, len(sites)) + 1)), replace=False)
            for site in active_sites:
                clicks = int(max(1, rng.poisson(lam=8 * engagement)))
                student_vle.append({
                    "code_module": mod,
                    "code_presentation": pres,
                    "id_student": sid,
                    "id_site": int(site),
                    "date": int(day + rng.integers(0, 6)),
                    "sum_click": clicks
                })

    pd.DataFrame(student_info).to_csv(output_dir / "studentInfo.csv", index=False)
    pd.DataFrame(student_reg).to_csv(output_dir / "studentRegistration.csv", index=False)
    pd.DataFrame(student_assessment).to_csv(output_dir / "studentAssessment.csv", index=False)
    pd.DataFrame(student_vle).to_csv(output_dir / "studentVle.csv", index=False)
    courses_df.to_csv(output_dir / "courses.csv", index=False)
    assessments_df.to_csv(output_dir / "assessments.csv", index=False)
    vle_df.to_csv(output_dir / "vle.csv", index=False)
