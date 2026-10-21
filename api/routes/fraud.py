"""
PlacementAI - Fraud Detection Routes
POST /fraud/flag
"""

import os, io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas import FraudCheckResponse, FraudFlag
from model.fraud_detector import check_fraud, check_fraud_batch
from api.routes.portfolio import _load_data

router = APIRouter()


@router.post("/fraud/flag")
async def fraud_flag(file: UploadFile = File(None), student_ids: list[str] = None):
    """
    Run fraud detection.
    - Upload a CSV file with student records, OR
    - Provide student_ids to check from the existing dataset
    """
    if file is not None:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        records = df.to_dict(orient="records")
    elif student_ids:
        full_df = _load_data()
        df = full_df[full_df["student_id"].isin(student_ids)]
        if df.empty:
            raise HTTPException(404, "No matching students found")
        records = df.to_dict(orient="records")
    else:
        # Check entire dataset
        full_df = _load_data()
        records = full_df.to_dict(orient="records")

    flagged = check_fraud_batch(records)

    results = []
    for item in flagged:
        results.append(FraudCheckResponse(
            student_id=item["student_id"],
            flags=[FraudFlag(**f) for f in item["flags"]],
        ))

    return {
        "total_checked": len(records),
        "total_flagged": len(results),
        "flagged_records": [r.model_dump() for r in results],
    }
