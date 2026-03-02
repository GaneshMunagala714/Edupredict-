"""
EduPredict Prediction Model
University AI Program Decision Predictor
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

MODELS_DIR = Path(__file__).parent
DB_PATH = MODELS_DIR / "predictions.db"


@dataclass
class UniversityProfile:
    university_type: str
    region: str
    current_cs_enrollment: int
    faculty_count: int
    budget_millions: float
    market_demand_score: float
    competition_level: str


@dataclass
class PredictionResult:
    recommendation: str
    confidence: float
    predicted_enrollment: int
    break_even_years: float
    roi_score: float
    key_factors: List[str]
    risk_factors: List[str]
    market_outlook: str


class SimpleDecisionTree:
    def predict(self, profile: UniversityProfile):
        score = 0.0
        factors = []
        risks = []
        
        # Market demand (25%)
        demand_score = profile.market_demand_score * 0.25
        score += demand_score
        if profile.market_demand_score >= 80:
            factors.append(f"Strong market demand: {profile.market_demand_score}/100")
        elif profile.market_demand_score < 50:
            risks.append(f"Weak market demand: {profile.market_demand_score}/100")
        
        # Budget (20%)
        budget_score = min(profile.budget_millions / 5, 20)
        score += budget_score
        if profile.budget_millions >= 10:
            factors.append(f"Strong budget: ${profile.budget_millions}M")
        elif profile.budget_millions < 3:
            risks.append(f"Limited budget: ${profile.budget_millions}M")
        
        # Faculty (15%)
        faculty_score = min(profile.faculty_count / 10, 15)
        score += faculty_score
        if profile.faculty_count >= 10:
            factors.append(f"Adequate faculty: {profile.faculty_count}")
        elif profile.faculty_count < 5:
            risks.append(f"Limited faculty: {profile.faculty_count}")
        
        # Competition (15%)
        comp_scores = {"low": 15, "medium": 8, "high": 3}
        score += comp_scores.get(profile.competition_level, 8)
        if profile.competition_level == "low":
            factors.append("Low competition in region")
        elif profile.competition_level == "high":
            risks.append("High competition from existing programs")
        
        # CS enrollment base (15%)
        score += min(profile.current_cs_enrollment / 100, 15)
        if profile.current_cs_enrollment >= 500:
            factors.append(f"Strong CS base: {profile.current_cs_enrollment} students")
        elif profile.current_cs_enrollment < 100:
            risks.append(f"Small CS base: {profile.current_cs_enrollment} students")
        
        # Type bonus (10%)
        type_scores = {"public": 8, "private": 6, "for_profit": 4}
        score += type_scores.get(profile.university_type, 6)
        
        confidence = min(0.95, 0.5 + (len(factors) * 0.1))
        
        if score >= 65:
            recommendation = "YES"
        elif score >= 45:
            recommendation = "MAYBE"
        else:
            recommendation = "NO"
        
        return recommendation, confidence, factors, risks, score


class EnrollmentPredictor:
    def predict(self, profile: UniversityProfile) -> int:
        base_rate = 0.15
        base_enrollment = int(profile.current_cs_enrollment * base_rate)
        market_mult = 0.8 + (profile.market_demand_score / 100) * 0.4
        comp_mults = {"low": 1.3, "medium": 1.0, "high": 0.7}
        comp_mult = comp_mults.get(profile.competition_level, 1.0)
        budget_mult = min(1.5, 0.8 + (profile.budget_millions / 20))
        predicted = int(base_enrollment * market_mult * comp_mult * budget_mult)
        return max(10, predicted)


class ROICalculator:
    def calculate(self, profile: UniversityProfile, enrollment: int):
        setup_cost = profile.budget_millions * 0.3
        annual_operating = profile.budget_millions * 0.2
        tuition = 25000
        annual_revenue = enrollment * tuition / 1_000_000
        
        cumulative = 0
        break_even = 0
        for year in range(1, 10):
            cumulative += annual_revenue - annual_operating
            if cumulative >= setup_cost and break_even == 0:
                break_even = year
                break
        if break_even == 0:
            break_even = 7
        
        roi = min(30, annual_revenue / profile.budget_millions * 30)
        roi += max(0, 25 - break_even * 3)
        roi += min(25, enrollment / 20)
        roi += min(20, profile.market_demand_score / 5)
        
        return break_even, roi, annual_revenue


class EduPredictModel:
    def __init__(self):
        self.decision_tree = SimpleDecisionTree()
        self.enrollment_predictor = EnrollmentPredictor()
        self.roi_calculator = ROICalculator()
        self.db_path = DB_PATH
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, university_type TEXT, region TEXT,
                cs_enrollment INTEGER, faculty_count INTEGER,
                budget_millions REAL, market_demand_score REAL,
                competition_level TEXT, recommendation TEXT,
                confidence REAL, predicted_enrollment INTEGER,
                break_even_years REAL, roi_score REAL,
                key_factors TEXT, risk_factors TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def predict(self, profile: UniversityProfile) -> PredictionResult:
        rec, conf, factors, risks, _ = self.decision_tree.predict(profile)
        predicted = self.enrollment_predictor.predict(profile)
        break_even, roi, _ = self.roi_calculator.calculate(profile, predicted)
        
        outlook = "Strong growth" if profile.market_demand_score >= 75 else \
                  "Moderate growth" if profile.market_demand_score >= 50 else \
                  "Saturated market"
        
        self._save_prediction(profile, rec, conf, predicted, break_even, roi, factors, risks)
        
        return PredictionResult(
            recommendation=rec, confidence=conf, predicted_enrollment=predicted,
            break_even_years=break_even, roi_score=roi, key_factors=factors,
            risk_factors=risks, market_outlook=outlook
        )
    
    def _save_prediction(self, profile, rec, conf, enrollment, break_even, roi, factors, risks):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), profile.university_type, profile.region,
              profile.current_cs_enrollment, profile.faculty_count,
              profile.budget_millions, profile.market_demand_score,
              profile.competition_level, rec, conf, enrollment,
              break_even, roi, json.dumps(factors), json.dumps(risks)))
        conn.commit()
        conn.close()
    
    def get_prediction_history(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_statistics(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT recommendation, COUNT(*) FROM predictions GROUP BY recommendation")
        recs = {r[0]: r[1] for r in cursor.fetchall()}
        cursor.execute("SELECT AVG(confidence), AVG(predicted_enrollment) FROM predictions")
        avg_conf, avg_enroll = cursor.fetchone()
        conn.close()
        return {"total_predictions": total, "recommendations": recs,
                "avg_confidence": avg_conf or 0, "avg_predicted_enrollment": avg_enroll or 0}


def create_model():
    return EduPredictModel()


def predict_from_dict(data: Dict) -> PredictionResult:
    profile = UniversityProfile(
        university_type=data.get("university_type", "public"),
        region=data.get("region", "unknown"),
        current_cs_enrollment=data.get("current_cs_enrollment", 200),
        faculty_count=data.get("faculty_count", 5),
        budget_millions=data.get("budget_millions", 5.0),
        market_demand_score=data.get("market_demand_score", 50.0),
        competition_level=data.get("competition_level", "medium")
    )
    return create_model().predict(profile)


if __name__ == "__main__":
    print("EduPredict Model Test")
    model = create_model()
    test = UniversityProfile("public", "Northeast", 800, 15, 15.0, 85, "low")
    result = model.predict(test)
    print(f"Rec: {result.recommendation}, Confidence: {result.confidence:.1%}")
    print(f"Enrollment: {result.predicted_enrollment}, Break-even: {result.break_even_years:.1f} years")
    print(f"ROI: {result.roi_score:.1f}/100, Outlook: {result.market_outlook}")
