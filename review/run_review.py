"""
run_review.py - Automated ACL 2027 Peer Review Pipeline via Claude Code CLI.
Invokes Claude with Sonnet 5.5 / latest to review manuscript/main.tex,
checks for mathematical consistency, empirical soundness, and citation integrity.
"""

import os
import subprocess
import sys
import shutil

def run_manuscript_review(tex_path: str = "manuscript/main.tex", out_review_path: str = "review/claude_review_report.md"):
    os.makedirs(os.path.dirname(out_review_path), exist_ok=True)
    
    claude_bin = shutil.which("claude") or r"C:\Users\vinh.dq4\.local\bin\claude.EXE"
    if not os.path.exists(claude_bin) and not shutil.which("claude"):
        print(f"Error: Claude CLI not found at {claude_bin}")
        return False
        
    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()
        
    review_prompt = f"""You are an Expert Senior Program Committee (SPC) Meta-Reviewer for the ACL 2027 Main Track.
Conduct a rigorous, critical, and constructive peer review of the following LaTeX manuscript on Preserving Invariant Ethical Topology in LLMs under Adversarial Injections.

Manuscript Content:
```latex
{tex_content}
```

Provide your formal review structured as follows:
1. SUMMARY OF CONTRIBUTIONS: High-level overview of theoretical and empirical claims.
2. STRENGTHS: Specific technical strengths, mathematical depth, novelty of ARETE, and experimental evaluation.
3. WEAKNESSES & PITFALLS: Critical analysis of assumptions, theorem formulations, potential failure modes, or missing baselines.
4. DETAILED RECOMMENDATIONS FOR AUTHORS: Specific LaTeX modifications, text additions, and clarifications to make the manuscript camera-ready.
5. PREDICTED SCORE: Standard ACL scale (1-5, with Confidence 1-5).
"""

    print("=== Submitting Manuscript to Claude Sonnet Reviewer via CLI ===")
    cmd = [claude_bin, "-p", review_prompt]
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180
        )
        if proc.returncode != 0:
            print(f"Claude review command returned code {proc.returncode}:\n{proc.stderr}")
            return False
            
        review_text = proc.stdout
        with open(out_review_path, "w", encoding="utf-8") as f:
            f.write("# ACL 2027 Peer Review Report (Claude Sonnet 5.5)\n\n")
            f.write(review_text)
        print(f"Review successfully generated and saved to {out_review_path}")
        return True
    except Exception as e:
        print(f"Exception while running Claude CLI: {e}")
        return False

if __name__ == "__main__":
    run_manuscript_review()
