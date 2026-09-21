r"""Build eval/dataset.jsonl from hand-written questions and verify every gold fact
appears (whitespace/case-normalised) on at least one of its gold pages of the PDF.

Run:  .\.venv\Scripts\python.exe eval\build_dataset.py
Page numbers are PDF page indices (1-based), which equal the "Page N of 524" header.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "raw" / "ewu_bulletin.pdf"
OUT = ROOT / "eval" / "dataset.jsonl"

# (category, question, gold_pages, gold_answer, gold_facts, extras)
# extras: ctx=[(role, text)...] for follow-ups, distractor=[pages] for near-miss questions
Q = []


def add(cat, q, pages, ans, facts, **kw):
    Q.append((cat, q, pages, ans, facts, kw))


# ---- single-fact (25)
S = "single_fact"
add(S, "What is the minimum attendance required in lectures, tutorials and practical classes?", [179], "Not less than 80%.", ["not less than 80%"])
add(S, "How much do I pay to fill in the online admission form?", [176], "Tk. 1,000.", ["paying Tk. 1, 000"])
add(S, "What is the one-time admission fee?", [179], "Tk. 15,000, non-refundable.", ["Tk.15, 000/-"])
add(S, "What is the admission fee for non-degree students?", [179], "Tk. 5,000.", ["Tk. 5, 000 is applicable for Non-Degree"])
add(S, "What is the re-admission fee if I change my degree program?", [179], "Tk. 15,000.", ["Re-admission fee of TK. 15,000"])
add(S, "What is the late registration fee?", [181, 215], "Tk. 500 to Tk. 1,000.", ["Tk. 500 to Tk.1, 000"])
add(S, "What is the maximum time allowed to complete an undergraduate degree on compassionate ground?", [25], "Seven (7) years.", ["Seven (7) years maximum"])
add(S, "What share of courses must be completed at EWU under the residency requirement?", [25], "At least 75%.", ["at least 75% of courses must be completed at EWU"])
add(S, "What percentage of credit hours can be accepted through credit transfer?", [214], "A maximum of 25%.", ["twenty five percent (25%)"])
add(S, "How much is the re-scrutiny fee for one final exam script?", [216], "A security deposit of Tk.200.", ["Tk.200 will be charged"])
add(S, "When is the library open on Sunday to Thursday?", [208], "8:30 am to 10:00 pm.", ["Sunday to Thursday"])
add(S, "How many clubs does EWU have?", [212], "18 clubs.", ["18 clubs"])
add(S, "How many credits are required for the BBA degree?", [23, 24], "123 credits.", ["a minimum of 123 credits"])
add(S, "What fee is charged for the remedial English course?", [180], "Tk 3,163 for one semester.", ["Tk 3, 163"])
add(S, "Who is the chairperson of the Department of Computer Science and Engineering?", [22], "Dr. Taskeed Jabid.", ["Department of Computer Science and Engineering Dr. Taskeed Jabid"])
add(S, "Who is the Dean of the Faculty of Sciences and Engineering?", [22], "Professor Dr. Mohamed Ruhul Amin.", ["Professor Dr. Mohamed Ruhul Amin"])
add(S, "What is the fee for verifying each academic document?", [177], "Tk. 50.00 (first copy free).", ["Tk. 50.00"])
add(S, "What SAT score qualifies for an admission test waiver?", [177], "A minimum total score of 1000.", ["minimum total score of 1000 in SAT"])
add(S, "What is the fee for an urgent official transcript?", [181], "Tk. 700.00.", ["For urgent: Tk. 700.00"])
add(S, "What is EWU's policy on cheating?", [218], "Zero tolerance.", ["zero tolerance on cheating"])
add(S, "For how long can a leave of absence be granted?", [218], "Up to three consecutive semesters / one academic year.", ["three consecutive semesters"])
add(S, "What does a duplicate degree certificate cost?", [181], "Tk. 1,000.00.", ["Duplicate Copy of Degree Certificate Fee: Tk. 1, 000.00"])
add(S, "What is the student activity fee per semester?", [180], "Tk. 510 (Tk. 765 for B.Pharm).", ["Tk. 510/-"])
add(S, "How much is the admission form fee for foreign students?", [176], "$13.00.", ["$13.00"])
add(S, "What is the minimum number of credits an undergraduate must register each semester?", [215], "3 courses (9 credits), except Pharmacy.", ["minimum of 3 (three) courses (9 credits)"])

# ---- exact-number (15)
N = "exact_number"
add(N, "What is the tuition fee per credit for the BBA program?", [179], "Tk. 5,500.", ["BBA 5, 500/-"])
add(N, "What is the tuition fee per credit for B.Sc. in CSE?", [179], "Tk. 5,500.", ["B. Sc. in CSE 5, 500/-"])
add(N, "What is the per-credit tuition for BA in English?", [179], "Tk. 4,000.", ["BA in English 4, 000/-"])
add(N, "What is the per-credit tuition for B.Pharm?", [180], "Tk. 6,000.", ["B. Pharm. 6, 000/-"])
add(N, "How much is the lab fee for B.Sc. in Civil Engineering per semester?", [180], "Tk. 2,650.", ["Tk. 2, 650/-"])
add(N, "How much is the lab fee for Bachelor of Pharmacy per semester?", [180], "Tk. 3,750.", ["Tk. 3, 750/-"])
add(N, "How many total credits does the B.Sc. in Civil Engineering need?", [24], "156.5 credits.", ["156.5 Credits"])
add(N, "How many credits is the Bachelor of Pharmacy program?", [24], "158 credits.", ["158 Credits"])
add(N, "How many credits does the LL.B (Hons.) require?", [23], "135 credits.", ["135 Credits"])
add(N, "What is the fee for correction of name?", [181], "Tk. 300.00.", ["Correction of Name: Tk. 300.00"])
add(N, "What is the credit transfer or waiver fee per credit?", [181], "Tk. 500.00 per credit.", ["Tk. 500.00 per credit"])
add(N, "How many credits of tuition waiver can a Trustee member award each semester?", [222], "Up to 27 credits.", ["upto 27 credits tuition fee waiver"])
add(N, "What minimum CGPA keeps the District Quota scholarship?", [220], "2.60.", ["minimum CGPA of 2.60"])
add(N, "How many students received scholarships in 2018-2019?", [219], "1921 students.", ["1921 students"])
add(N, "What is the registration fee for the Pharmacy Council?", [180], "Tk. 300.00.", ["Tk. 300.00"])

# ---- table-sourced (15)
T = "table"
add(T, "What grade point does an A- carry?", [216], "3.70.", ["87 - below 90 A- 3.70"])
add(T, "What numerical score range earns a B+?", [216], "83 to below 87.", ["83 - below 87 B+ 3.30"])
add(T, "What grade point is a D worth?", [216], "1.00.", ["60 - below 63 D 1.00"])
add(T, "What grade point does a C+ have?", [216], "2.30.", ["73 - below 77 C+ 2.30"])
add(T, "Which letter grades are passing grades?", [216], "A, B, C and D; F is failing.", ["A, B, C, and D are considered passing grades"])
add(T, "How many credits must a CSE undergraduate complete in the last year to be eligible for merit scholarship?", [221], "35 credits.", ["B.Sc. in Computer Science & Engineering 35"])
add(T, "How many credits must a B.Pharm student earn to be eligible for merit scholarship?", [221], "39 credits.", ["Bachelor of Pharmacy 39"])
add(T, "How many credits must an MS in CSE student have completed in two semesters for scholarship eligibility?", [222], "18 credits.", ["MS in CSE 2 18"])
add(T, "What are the prerequisites of Data Structures (CSE207)?", [116], "CSE110.", ["CSE207 Data Structures 3+1=4 CSE110"])
add(T, "What is the prerequisite of the Algorithms course CSE326?", [116], "CSE207.", ["CSE326 Algorithms 3+1.5=4.5 CSE207"])
add(T, "How many credits is Object Oriented Programming in the CSE curriculum?", [116], "3+1.5 = 4.5 credits.", ["CSE110 Object Oriented Programming 3+1.5=4.5"])
add(T, "How many credit hours must a CSE student complete before starting the Capstone Project?", [117], "At least 105 credit hours.", ["Completed at least 105 credit hours"])
add(T, "How many credits do the core mathematics and statistics courses total for CSE?", [115], "15 credits.", ["Core Mathematics and Statistics Course 15"])
add(T, "How many credits do Core Natural Science courses total in the CSE program?", [115], "11 credits (9+2).", ["Core Natural Science Courses 9+2=11"])
add(T, "What is the semester credit requirement for MBA students to qualify for the merit scholarship?", [221], "29 credits in 3 semesters.", ["MBA 3 29"])

# ---- program-specific (15)
PG = "program_specific"
add(PG, "How many credits does the B.Sc. in CSE require?", [24, 114], "140 credits.", ["Total 140"])
add(PG, "How many credits is the B.Sc. in EEE program?", [24], "140 credits.", ["Electrical and Electronic Engineering (EEE) - 140 Credits"])
add(PG, "How many credits is the B.Sc. in Genetic Engineering and Biotechnology?", [24], "133 credits.", ["Biotechnology -133 Credits"])
add(PG, "How many credits is the BS in Applied Statistics?", [24], "127 credits.", ["Applied Statistics- 127 Credits"])
add(PG, "Which courses are in Group B of the minor in Computer Science and Engineering?", [26], "CSE 411, CSE 348, CSE 442, CSE 480 (any two).", ["Group B: Any two from the following courses: CSE 411, CSE 348, CSE 442, CSE 480"])
add(PG, "Which courses can be chosen for Group A of the CSE minor?", [26], "Any five of CSE 105, 107, 207, 245, 301, 209, 251, 345.", ["CSE 105, CSE 107, CSE 207, CSE 245, CSE 301, CSE 209, CSE 251, CSE 345"])
add(PG, "Which courses are in Group B of the EEE minor?", [26], "Any two of EEE 401, 403, 416, 423, 445.", ["EEE 401, EEE 403, EEE 416, EEE 423, EEE 445"])
add(PG, "Which courses are in Group B of the ETE minor?", [26], "Any two of ETE 401, 403, 430, 441, 442, 444.", ["ETE 401, ETE 403, ETE 430, ETE 441, ETE 442, ETE 444"])
add(PG, "How many credits are needed for a second major in Economics for non-Economics students?", [49], "45 credits.", ["Total Credit Requirement 45"])
add(PG, "What is the credit requirement for the Bachelor of Social Science in Sociology?", [76], "123 credits.", ["Sociology Requirement: 123 Credits"])
add(PG, "How many elective courses must a Sociology student choose?", [78], "11 courses (33 credits).", ["choose any 11 courses"])
add(PG, "What are the compulsory language and general education courses in CSE?", [114], "ENG101, ENG102 and GEN226 (9 credits).", ["GEN226 Emergence of Bangladesh 3"])
add(PG, "How many major elective courses does a CSE student take in the chosen major area?", [117], "Three elective courses (9+3=12 credits) plus two compulsory.", ["Three elective courses (9+3=12 credits)"])
add(PG, "How many non-major elective credits does CSE require?", [117], "Minimum 8 credits.", ["Non-Major Elective Requirements Minimum 8 credits"])
add(PG, "What are the minimum HSC grades in Chemistry and Biology for Pharmacy admission?", [143], "GPA 3.0 in Chemistry and Biology separately and 2.0 in Mathematics.", ["minimum GPA 3.0 in Chemistry and Biology separately"])

# ---- multi-hop (10)
M = "multi_hop"
add(M, "If I retake a course, can I still get a gold medal or a scholarship?", [26, 218, 219], "No: retake makes a student ineligible for both.", ["shall not be eligible for getting Gold Medal", "Students availing the advantage of retaking any course any time will not be eligible for any scholarship"])
add(M, "If I am on academic probation, can I take a leave of absence?", [217, 218], "No: students on probation cannot drop a semester or take leave (with limited exceptions).", ["not allowed to drop a semester or to take leave of absence"])
add(M, "What is the maximum credit share I can transfer and what does credit transfer cost per credit?", [214, 181], "25% of credits; Tk. 500 per credit.", ["twenty five percent (25%)", "Tk. 500.00 per credit"])
add(M, "What CGPA do I need to graduate and what happens if my CGPA falls below 2?", [25, 217], "Minimum 2.00; below 2 leads to probation.", ["minimum CGPA of 2.00", "will again be placed on probation"])
add(M, "For a CSE student, what are the lab fee and student activity fee per semester?", [180], "Tk. 2,500 lab and Tk. 510 activity fee.", ["Tk. 2, 500/-(non refundable) per semester for CSE", "Tk. 510/-"])
add(M, "What are the requirements for the freedom fighter admission quota and tuition waiver?", [223], "3% admission quota, up to 100% waiver.", ["3% admission quota", "maximum 100% tuition waiver"])
add(M, "Which programs need Math competence and what SAT-waiver GPA do Science and Engineering applicants need?", [176, 177], "CSE/EEE/ICE/ETE need HSC Math; FSE waiver needs GPA 3.5 in Math and Physics with SAT 1000.", ["competence in HSC-level Mathematics", "minimum GPA of 3.5 in Math and Physics"])
add(M, "If I fail the remedial English course once, what do I pay next time?", [180, 217], "Regular course fee of Tk.12,000; failing twice leads to probation.", ["regular course fees of Tk.12,000", "fail to pass in remedial courses in two attempts"])
add(M, "Can a B.Pharm student on a merit scholarship register fewer than 12 credits per semester?", [215, 220], "No: B.Pharm must register at least 12 credits.", ["12 credits in a semester for the students of B.Pharm"])
add(M, "How is CGPA computed for repeated courses and what happens to my earlier F?", [217, 218], "Last attempt counts; earlier F converted to R.", ["last attempt of the course(s) only", "will be converted to"])

# ---- comparison (10)
C = "comparison"
add(C, "How does the per-credit tuition for BBA compare with B.Pharm?", [179, 180], "BBA Tk. 5,500 vs B.Pharm Tk. 6,000.", ["BBA 5, 500/-", "B. Pharm. 6, 000/-"])
add(C, "Compare the lab fees for CSE, Civil Engineering and Pharmacy.", [180], "CSE Tk. 2,500; Civil Tk. 2,650; Pharmacy Tk. 3,750.", ["Tk. 2, 650/-", "Tk. 3, 750/-"])
add(C, "Which needs more credits, CSE or Civil Engineering?", [24], "Civil (156.5) needs more than CSE (140).", ["156.5 Credits", "Computer Science and Engineering (CSE) - 140 Credits"])
add(C, "What is the difference between Summa Cum Laude and Magna Cum Laude CGPA requirements?", [26], "Summa 3.90 or above; Magna 3.80 to less than 3.90.", ["CGPA of 3.90 or above", "CGPA of 3.80 to less than 3.90"])
add(C, "How does the student activity fee differ between B.Pharm and other departments?", [180], "Tk. 765 for B.Pharm, Tk. 510 for others.", ["Tk. 765/-", "Tk. 510/-"])
add(C, "How does tuition for BBA compare with BSS in Economics?", [179], "BBA Tk. 5,500; Economics Tk. 4,000.", ["BBA 5, 500/-", "BSS in Economics 4, 000/-"])
add(C, "What is the gold medal CGPA for undergraduates versus graduate students?", [26], "4.00 undergraduate; 3.99 graduate.", ["CGPA of 4.00 or above", "CGPA of 3.99 or above"])
add(C, "What CGPA leads to probation versus automatic dismissal?", [217], "Between 1 and 2: probation; below 1.0: dismissal.", ["CGPA will be between 1 and 2", "falls below 1.0"])
add(C, "How do library hours on Friday differ from Sunday-Thursday?", [208], "Friday 8:30 am to 5:00 pm (1-2 pm break); Sun-Thu to 10:00 pm.", ["Friday : 08:30 am"])
add(C, "Compare merit-scholarship credit requirements for CSE and Bachelor of Laws.", [221], "CSE 35; LL.B. 33.", ["Bachelor of Laws [LL. B. (Hons.)] 33", "B.Sc. in Computer Science & Engineering 35"])

# ---- follow-up (10): the retrieval query is the bare follow-up; context carries the antecedent
F = "follow_up"
add(F, "And for CSE?", [179], "Tk. 5,500 per credit.", ["B. Sc. in CSE 5, 500/-"], ctx=[("user", "What is the tuition per credit for BBA?"), ("assistant", "BBA tuition is Tk. 5,500 per credit. [Page 179]")])
add(F, "What about B.Pharm?", [180], "Tk. 6,000 per credit.", ["B. Pharm. 6, 000/-"], ctx=[("user", "What is the tuition per credit for CSE?"), ("assistant", "Tk. 5,500 per credit. [Page 179]")])
add(F, "Is there also a lab fee for it?", [180], "Yes, Tk. 2,500 per semester for CSE.", ["Tk. 2, 500/-(non refundable) per semester for CSE"], ctx=[("user", "What is the tuition per credit for CSE?"), ("assistant", "Tk. 5,500 per credit. [Page 179]")])
add(F, "What happens if I don't raise it in time?", [217], "Failure to raise CGPA to at least 2 leads to dismissal.", ["will lead to dismissal from the university"], ctx=[("user", "What is academic probation?"), ("assistant", "A CGPA between 1 and 2 places you on probation for the next two semesters. [Page 217]")])
add(F, "Can I still get a scholarship if I do that?", [219], "No: retaking makes you ineligible for scholarships.", ["will not be eligible for any scholarship"], ctx=[("user", "Can I retake a course?"), ("assistant", "Yes, a particular course only once with any earlier grade. [Page 217]")])
add(F, "How many credits is that program?", [24, 114], "140 credits.", ["Total 140"], ctx=[("user", "Which department chairperson leads CSE?"), ("assistant", "Dr. Taskeed Jabid chairs Computer Science and Engineering. [Page 22]")])
add(F, "What about for graduate students?", [26], "3.99 or above for graduate gold medal.", ["CGPA of 3.99 or above"], ctx=[("user", "What CGPA gives an undergraduate a gold medal?"), ("assistant", "A CGPA of 4.00 or above within four years. [Page 26]")])
add(F, "And how much is the urgent one?", [181], "Tk. 700.00.", ["For urgent: Tk. 700.00"], ctx=[("user", "What is the official transcript fee?"), ("assistant", "Tk. 500.00. [Page 181]")])
add(F, "And on Fridays?", [208], "8:30 am to 5:00 pm with a 1-2 pm break.", ["Friday : 08:30 am"], ctx=[("user", "What are the library hours from Sunday to Thursday?"), ("assistant", "8:30 am to 10:00 pm. [Page 208]")])
add(F, "What is the minimum for Pharmacy students?", [220], "12 credits per semester.", ["12 credits in a semester for the students of B.Pharm"], ctx=[("user", "How many credits must I register each semester for a scholarship?"), ("assistant", "At least 9 credits. [Page 220]")])

# ---- unanswerable (15): absence of these topics in the PDF was verified with a text search
U = "unanswerable"
for uq in [
    "What IELTS score do I need to get admitted?",
    "Does EWU offer a Bachelor of Nursing program?",
    "Does EWU have a football team?",
    "What are the campus bus service timings?",
    "Can I pay my tuition in instalments?",
    "Does EWU have a student exchange program with a university in Europe?",
    "Can students take part-time jobs on campus?",
    "Does the university have a gym for students?",
    "Is there a daycare center on campus?",
    "Where is the university health center located?",
    "What is the deadline to apply for Fall 2025 admission?",
    "What is the placement rate of CSE graduates?",
    "Is there a swimming pool on campus?",
    "How much does a dormitory room cost per month?",
    "Where is the cricket ground?",
]:
    add(U, uq, [], "", [], answerable=False)

# ---- adversarial near-miss (10)
A = "adversarial"
add(A, "What GPA must I maintain for a scholarship?", [219, 220, 222], "Depends on the scheme: 3.50 for merit, 2.60 for District Quota/financial aid; not the admission GPA of 3.00.", ["minimum CGPA of 2.60"], distractor=[176])
add(A, "What is the minimum CGPA required to graduate?", [25], "2.00.", ["minimum CGPA of 2.00"], distractor=[217, 220])
add(A, "What is the fee for the admission form?", [176], "Tk. 1,000 (foreign students $13.00).", ["paying Tk. 1, 000"], distractor=[179])
add(A, "At what CGPA is a student automatically dismissed?", [217], "Below 1.0 after the first two or subsequent semesters.", ["falls below 1.0"], distractor=[220])
add(A, "What CGPA is required for the Medha Lalon Scholarship?", [224], "3.50.", ["CGPA requirement for awarding Medha Lalon Scholarship is 3.50"], distractor=[220])
add(A, "What percentage admission quota is reserved for freedom fighters wards?", [223], "3%.", ["3% admission quota"], distractor=[219])
add(A, "What are the library hours on Saturday?", [208], "5:00 pm to 10:00 pm.", ["Saturday : 5:00 pm"], distractor=[207])
add(A, "What is the grade point for an A+?", [216], "4.00.", ["97-100 A+ 4.00"], distractor=[26])
add(A, "What CGPA is needed for Cum Laude?", [26], "3.75 to less than 3.80.", ["CGPA of 3.75 to less than 3.80"], distractor=[220])
add(A, "How many credits are needed to graduate from the CSE department, not the MS in CSE?", [24], "140 credits (MS in CSE is 33).", ["Computer Science and Engineering (CSE) - 140 Credits"], distractor=[222])


def norm(s):
    return re.sub(r"\s+", " ", s).lower()


def main():
    doc = pymupdf.open(str(PDF))
    pages = {i + 1: norm(p.get_text()) for i, p in enumerate(doc)}
    rows, bad = [], 0
    for i, (cat, q, gp, ans, facts, kw) in enumerate(Q, 1):
        for f in facts:
            if not any(norm(f) in pages[p] for p in gp):
                print(f"FACT NOT FOUND q{i}: {f!r} pages {gp}")
                bad += 1
        row = {
            "qid": f"q{i:03d}",
            "question": q,
            "category": cat,
            "gold_pages": gp,
            "gold_answer": ans,
            "gold_facts": facts,
            "answerable": kw.get("answerable", True),
            "conversation_context": [{"role": r, "content": c} for r, c in kw.get("ctx", [])],
        }
        if "distractor" in kw:
            row["distractor_pages"] = kw["distractor"]
        rows.append(row)
    if bad:
        sys.exit(f"{bad} gold facts missing; fix before writing")
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(len(rows), "questions ->", OUT)
    print(dict(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
