# MOIC_Inventory
## Cấu trúc thư mục
thesis-ml/
├─ README.md
├─ requirements.txt  (hoặc pyproject.toml)
├─ .gitignore
├─ configs/
│  ├─ base.yaml
│  ├─ data.yaml
│  ├─ models/
│  │  ├─ lr.yaml
│  │  ├─ rf.yaml
│  │  └─ xgb.yaml
│  └─ experiments/
│     ├─ exp01_baseline.yaml
│     └─ exp02_tuning.yaml
├─ data/
│  ├─ raw/            (dữ liệu gốc, KHÔNG sửa)
│  ├─ interim/        (dữ liệu trung gian)
│  ├─ processed/      (dataset final cho modeling)
│  └─ external/       (nguồn ngoài)
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature.ipynb
│  └─ 03_results.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ paths.py        (quy ước đường dẫn)
│  ├─ utils/
│  │  ├─ seed.py
│  │  ├─ logger.py
│  │  └─ io.py
│  ├─ data/
│  │  ├─ make_dataset.py   (raw → processed)
│  │  ├─ split.py
│  │  └─ validation.py
│  ├─ features/
│  │  ├─ build_features.py
│  │  └─ encoders.py
│  ├─ models/
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  ├─ calibrate.py
│  │  └─ explain.py        (SHAP/LIME nếu có)
│  └─ pipelines/
│     ├─ run_train.py
│     └─ run_infer.py
├─ reports/
│  ├─ figures/        (hình cho luận văn)
│  ├─ tables/         (bảng cho luận văn)
│  └─ draft_notes/    (ghi chú kết quả)
├─ results/
│  ├─ runs/
│  │  └─ 2026-02-25_exp01/
│  │     ├─ config_used.yaml
│  │     ├─ metrics.json
│  │     ├─ model.pkl
│  │     └─ preds.csv
│  └─ model_registry.csv   (tổng hợp các run)
└─ tests/             (tối thiểu cho các hàm data/features)

## Trình tự tạo cấu trúc thư mục
**Bước 1:** Tạo thư mục gốc lớn nhất

**Bước 2:** Tạo cấu trúc thư mục lớn hơn
Ví dụ: mkdir configs data notebooks src reports results tests

**Bước 3:** Tạo các thư mục con quan trọng tuỳ thuộc vào nhu cầu của người dùng
mkdir configs/models configs/experiments
mkdir data/raw data/interim data/processed data/external
mkdir src/utils src/data src/features src/models src/pipelines
mkdir reports/figures reports/tables reports/draft_notes
mkdir results/runs

Sau đó tạo file __init__.py trong src

**Bước 4** Tạo ra các thư mục con

## Giải thích ý nghĩa một số thư mục con
| File      | Giải quyết vấn đề            | Ảnh hưởng tới thesis |
| --------- | ---------------------------- | -------------------- |
| seed.py   | Kết quả không ổn định        | Tính tái tạo         |
| logger.py | Không truy vết được training | Minh bạch            |
| io.py     | Quản lý model lộn xộn        | Tổ chức thí nghiệm   |
