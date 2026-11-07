# Submission / hand-in checklist

Use this when packaging the capstone for coordinators.

## Artefacts

- [ ] GitHub repo shared / tagged (code + docs; no `.env`)  
- [ ] Hugging Face model `neghani/plantvillage-mobilenetv2` with tags `v1.0`, `v1.1`, `v1.2`  
- [ ] Model card on Hub has lab metrics + limitations  
- [ ] `docs/` pack present (see `docs/README.md`)  
- [ ] Proposal doc retained for comparison (`Plant_Disease_Detection_Proposal_v1.docx`)  
- [ ] `model/final_report.md` present (locked field numbers)  
- [ ] Figma link in root README  

## Claims aligned

- [ ] Live pin = MobileNet `@v1.2` everywhere you speak / write  
- [ ] Field number = **137/284 (48.2%)** from `@v1.1`  
- [ ] Lab vs field not mixed  
- [ ] ResNet only as bake-off comparison  
- [ ] LLM never sees the image — stated in overview + architecture  
- [ ] Telugu shipped; Hindi not claimed as done  
- [ ] 48.2% &lt; 50% stated openly  

## Demo readiness

- [ ] `phase2-demo-checklist.md` green on demo laptop  
- [ ] `demo-day-prep-guide.md` pitch + Q&A rehearsed  
- [ ] Deck has **training story** slides (`09-training-story.md` + `assets/charts/`) — not a lab→field jump  
- [ ] Deck has **product screenshots** (`assets/screenshots/03-results-ok.png` at minimum)  
- [ ] Backup screen recording  
- [ ] Demo image set folder ready  

## Code hygiene

- [ ] `shared/release.json` == `hf_space/release.json`  
- [ ] No secrets in git  
- [ ] `PYTHONPATH=..:. pytest -q` green in api venv  
- [ ] Root README matches MobileNet pin (not stale ResNet text)
