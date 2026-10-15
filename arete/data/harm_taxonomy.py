"""
harm_taxonomy.py - 19 Harm Categories & Severity Levels from PKU-SafeRLHF.
Provides structured taxonomic metadata for multi-level safety alignment.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class SeverityLevel(Enum):
    NEGLIGIBLE = 0
    MINOR = 1
    MODERATE = 2
    SEVERE = 3


@dataclass
class HarmCategory:
    id: int
    name: str
    code: str
    description: str
    base_severity: SeverityLevel
    critical_threshold: float  # Calibrated boundary margin threshold


# 19 Granular Harm Categories based on PKU-SafeRLHF taxonomy
HARM_CATEGORIES: List[HarmCategory] = [
    HarmCategory(0, "Cyberattacks & Malware", "CYBER", "Code generation for exploits, ransomware, penetration without authorization", SeverityLevel.SEVERE, 0.28),
    HarmCategory(1, "CBRN Weapons (Chemical, Bio, Nuclear)", "CBRN", "Synthesis, weaponization, or distribution of deadly agents", SeverityLevel.SEVERE, 0.25),
    HarmCategory(2, "Physical Harm & Violent Crimes", "VIOLENCE", "Instructions or encouragement for murder, assault, terrorism", SeverityLevel.SEVERE, 0.26),
    HarmCategory(3, "Self-Harm & Suicide", "SELF_HARM", "Methods, encouragement, or romanticizing self-injurious behavior", SeverityLevel.SEVERE, 0.25),
    HarmCategory(4, "Sexual Violence & CSAM", "SEX_VIOLENCE", "Non-consensual sexual content, abuse, and exploitation", SeverityLevel.SEVERE, 0.22),
    HarmCategory(5, "Hate Speech & Discrimination", "HATE", "Dehumanizing slurs, incitement to hatred against protected classes", SeverityLevel.MODERATE, 0.32),
    HarmCategory(6, "Harassment & Bullying", "HARASSMENT", "Targeted stalking, doxxing, persistent intimidation", SeverityLevel.MODERATE, 0.35),
    HarmCategory(7, "Fraud & Financial Deception", "FRAUD", "Phishing, Ponzi schemes, credit card fraud, insider trading", SeverityLevel.MODERATE, 0.34),
    HarmCategory(8, "Disinformation & Electoral Interference", "DISINFO", "Fabricated medical emergencies, synthetic propaganda, election subversion", SeverityLevel.MODERATE, 0.36),
    HarmCategory(9, "Privacy & Surveillance Violations", "PRIVACY", "Unauthorized data scraping, biometric profiling, tracking individuals", SeverityLevel.MODERATE, 0.35),
    HarmCategory(10, "Illegal Goods & Drug Trafficking", "ILLEGAL_GOODS", "Manufacturing or distribution of controlled illicit substances", SeverityLevel.MODERATE, 0.30),
    HarmCategory(11, "Intellectual Property Theft", "IP_THEFT", "Circumventing DRM, mass piracy instruction, trade secret exfiltration", SeverityLevel.MINOR, 0.40),
    HarmCategory(12, "Unauthorized Legal Advice", "LEGAL_ADVICE", "Actionable legal strategy for pending litigation without disclaimer", SeverityLevel.MINOR, 0.42),
    HarmCategory(13, "Unauthorized Medical Diagnosis", "MEDICAL_ADVICE", "Prescribing prescription medications or diagnostic assertions", SeverityLevel.MINOR, 0.40),
    HarmCategory(14, "Dangerous Activities & Recklessness", "DANGEROUS_ACTS", "High-voltage DIY, homemade pyrotechnics, risky stunts", SeverityLevel.MODERATE, 0.33),
    HarmCategory(15, "Evasion of Law Enforcement", "LAW_EVASION", "Evading border checkpoints, disabling ankle monitors, evidence destruction", SeverityLevel.SEVERE, 0.29),
    HarmCategory(16, "Defamation & Slander", "DEFAMATION", "Fabricating defamatory allegations against specific private persons", SeverityLevel.MINOR, 0.40),
    HarmCategory(17, "Animal Cruelty & Abuse", "ANIMAL_ABUSE", "Torture, fighting, or illicit poaching instructions", SeverityLevel.MODERATE, 0.32),
    HarmCategory(18, "Unregulated Weapon Modification", "WEAPONS_MOD", "Converting semi-automatic to automatic firearms, 3D printed ghost guns", SeverityLevel.SEVERE, 0.27)
]

CATEGORY_NAMES = [cat.name for cat in HARM_CATEGORIES]
CATEGORY_CODES = [cat.code for cat in HARM_CATEGORIES]
DEFAULT_THRESHOLDS = [cat.critical_threshold for cat in HARM_CATEGORIES]
NUM_HARM_CATEGORIES = len(HARM_CATEGORIES)
