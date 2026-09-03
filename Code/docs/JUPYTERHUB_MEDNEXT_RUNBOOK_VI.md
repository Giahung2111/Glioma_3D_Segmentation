# Hướng dẫn chạy MedNeXt hoàn chỉnh trên JupyterHub

Tài liệu này dành cho máy JupyterHub có RTX 3090 24 GB. Mục tiêu là chạy đúng
pipeline MedNeXt-S-k3 của project từ dữ liệu raw đến report, không sửa source
MedNeXt chính thức và không cài package vào Python dùng chung của JupyterHub.

## Nguyên tắc bắt buộc

- Git repository thật nằm tại
  `/var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation`.
- `Datasets` và `Workspace` phải nằm trên ổ bền vững `/mnt/d`, sau đó được liên
  kết vào repository bằng symbolic link.
- Hai môi trường Python riêng nằm trong `Code/.venv-models` và
  `Code/.venv-brats-metrics` trên filesystem Linux.
- Luôn hoàn thành smoke test 3 epoch trước. Pipeline full sẽ tự từ chối chạy nếu
  smoke test, resume, inference, metrics, failure analysis và report chưa pass.
- Không dùng `sudo`. Không tự chọn batch size. Batch size lấy từ official
  MedNeXt planner và được kiểm tra thực tế bằng GPU memory preflight.

## A. Cập nhật code từ máy local lên GitHub

Mở **PowerShell tại máy local**, tại thư mục
`C:\Projects\Glioma_3D_Segmentation`, rồi chạy lần lượt:

```powershell
cd C:\Projects\Glioma_3D_Segmentation
git status --short
git add .gitattributes .gitignore `
  Code/scripts/setup_research_models_env.sh `
  Code/scripts/run_mednext_cv_pipeline.sh `
  Code/src/glioma_seg/backends/mednext/backend.py `
  Code/src/glioma_seg/backends/mednext/trainer.py `
  Code/src/glioma_seg/backends/mednext/smoke_gate.py `
  Code/src/glioma_seg/data/canonical_splits.py `
  Code/src/glioma_seg/reporting/model_runtime.py `
  Code/tests/test_linux_mednext_pipeline.py `
  Code/docs/JUPYTERHUB_MEDNEXT_RUNBOOK_VI.md
git diff --cached --check
git commit -m "Add fail-closed Linux MedNeXt pipeline"
git push origin main
```

Không thêm `Datasets`, `Workspace`, môi trường hoặc checkpoint vào Git.

## B. Tạo vùng lưu trữ bền vững trên JupyterHub

Mở **Terminal trong JupyterHub**. Dán nguyên khối lệnh sau. Nếu repository đang
có thư mục `Datasets` hoặc `Workspace` thật trong `/var/tmp`, lệnh chỉ chuyển nó
sang thư mục backup trên `/mnt/d`; lệnh không xóa dữ liệu.

```bash
(
set -e
REPO=/var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
STORE=/mnt/d/Henry/_storage/Glioma_3D_Segmentation_data
STAMP=$(date +%Y%m%d_%H%M%S)

test -d "$REPO/.git"
mkdir -p "$STORE/Datasets" "$STORE/Workspace" "$STORE/backups"
cd "$REPO"

if [ -e Datasets ] && [ ! -L Datasets ]; then
  mv Datasets "$STORE/backups/Datasets_before_link_$STAMP"
fi
if [ -L Datasets ] && [ "$(readlink -f Datasets)" != "$STORE/Datasets" ]; then
  echo "STOP: Datasets đang trỏ tới nơi khác: $(readlink -f Datasets)"
  false
fi
if [ ! -L Datasets ]; then
  ln -s "$STORE/Datasets" Datasets
fi

if [ -e Workspace ] && [ ! -L Workspace ]; then
  mv Workspace "$STORE/backups/Workspace_before_link_$STAMP"
fi
if [ -L Workspace ] && [ "$(readlink -f Workspace)" != "$STORE/Workspace" ]; then
  echo "STOP: Workspace đang trỏ tới nơi khác: $(readlink -f Workspace)"
  false
fi
if [ ! -L Workspace ]; then
  ln -s "$STORE/Workspace" Workspace
fi

echo "Datasets -> $(readlink -f Datasets)"
echo "Workspace -> $(readlink -f Workspace)"
)
```

Kết quả đúng phải là:

```text
Datasets -> /mnt/d/Henry/_storage/Glioma_3D_Segmentation_data/Datasets
Workspace -> /mnt/d/Henry/_storage/Glioma_3D_Segmentation_data/Workspace
```

## C. Upload và kiểm tra dataset

Trong giao diện file JupyterHub, mở:

```text
Glioma_3D_Segmentation_code/Datasets
```

Upload nguyên thư mục:

```text
ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
```

Trong lúc upload bằng giao diện web, không reload/đóng tab và không để máy đang
mở trình duyệt sleep. Reload thường hủy các file chưa gửi; những file đã ghi
xong trên server vẫn được giữ lại. Nếu bị gián đoạn, không xóa thư mục partial:
upload lại chính thư mục nguồn đầy đủ vào cùng thư mục `Datasets`, chấp nhận
overwrite/merge các tên trùng nếu giao diện hỏi, rồi kiểm tra lại đủ 1.251 thư
mục và 6.255 file. Pipeline vẫn chạy validation nội dung trước khi training.

Vì `Datasets` là symbolic link, dữ liệu thật sẽ được ghi vào `/mnt/d`, không nằm
trong `/var/tmp`.

Ngay sau bước B, thư mục `Datasets` mới có thể đang trống. Lệnh
`TRAIN=/mnt/...` ở dưới **chỉ gán một biến đường dẫn**; nó không tạo thư mục và
không upload dữ liệu. Vì vậy, phải chờ upload xong và nhìn thấy thư mục
`ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData` bên trong `Datasets` rồi mới
chạy lệnh đếm.

Sau khi giao diện báo upload xong, chạy:

```bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
TRAIN="$PWD/Datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

echo -n "Cases: "
find "$TRAIN" -mindepth 1 -maxdepth 1 -type d -name 'BraTS-GLI-*' | wc -l

echo -n "NIfTI files: "
find "$TRAIN" -type f -name '*.nii.gz' | wc -l

du -sh "$TRAIN"
```

Chỉ đi tiếp khi hai số đầu là:

```text
Cases: 1251
NIfTI files: 6255
```

Nếu chưa đúng, upload chưa hoàn tất. Không chạy setup hoặc training.

## D. Pull code mới nhất và submodule

Trong **Terminal JupyterHub**, chạy:

```bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
git status --short
git pull --ff-only origin main
git submodule sync --recursive
git submodule update --init --recursive
git status --short
git submodule status
```

`git status --short` cuối cùng không được báo thay đổi trong `Code`. Dấu cách ở
đầu mỗi dòng `git submodule status` nghĩa là submodule đang đúng commit.

## E. Kiểm tra công cụ hệ thống trước khi cài

Chạy:

```bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
/opt/jupyterhub/bin/python --version
nvidia-smi
command -v git
command -v tmux
command -v python3.9 || command -v conda || command -v micromamba
```

Điều kiện đúng:

- Python chính là 3.10.x.
- `nvidia-smi` thấy RTX 3090 và khoảng 24576 MiB VRAM.
- Có đường dẫn cho `git` và `tmux`.
- Dòng cuối phải trả về Python 3.9, `conda`, hoặc `micromamba` để tạo môi trường
  tương thích cho official BraTS evaluator.

Nếu `tmux` hoặc cả ba lựa chọn ở dòng cuối đều không tồn tại, dừng tại đây và
nhờ quản trị viên cài/cấp công cụ ở mức user. Không dùng `sudo`.

## F. Tạo môi trường Python riêng

Chạy đúng một lệnh sau:

```bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
bash Code/scripts/setup_research_models_env.sh --python /opt/jupyterhub/bin/python
```

Lệnh này sẽ:

1. kiểm tra commit của MedNeXt, MONAI, MONAI model zoo và official BraTS metrics;
2. tạo `Code/.venv-models` bằng Python 3.10;
3. cài PyTorch 2.5.1 CUDA 12.1 và các dependency đã pin;
4. cài official MedNeXt ở editable mode, không sửa source upstream;
5. cài project `Code` ở editable mode;
6. tạo `Code/.venv-brats-metrics` bằng Python 3.9 cho official BraTS evaluator;
7. xác nhận CUDA và in tên RTX 3090.

Kết quả cuối đúng phải có:

```text
Research-model Linux environments are ready.
GPU=NVIDIA GeForce RTX 3090
```

Không cần chạy `source .../activate`. Hai pipeline script gọi thẳng đúng Python.

## G. Chạy smoke pipeline 3 epoch bắt buộc

Tạo phiên terminal bền vững:

```bash
tmux new -s mednext_smoke
```

Trong màn hình tmux vừa mở, chạy:

```bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
bash Code/scripts/run_mednext_cv_pipeline.sh --smoke-test --confirm-run
```

Smoke pipeline tự làm toàn bộ:

1. validate đủ 1.251 raw cases;
2. convert sang layout dùng chung và tạo đúng split 5-fold;
3. chạy test gate;
4. chạy official MedNeXt preprocessing trên 8 train + 2 validation cases;
5. chạy một bước forward/loss/backward thật để kiểm tra VRAM;
6. train epoch 1, cố ý dừng và audit checkpoint;
7. resume đúng checkpoint để train đến epoch 3;
8. inference và giữ mask, native NPZ, canonical ET/TC/WT NPZ;
9. tính semantic Dice/HD95 và official lesion-wise metrics;
10. chạy failure statistics/ranking;
11. tổng hợp GPU, runtime và inference time;
12. tạo `summary.md`, `weekly_discussion.md` và audit report bundle.

Trong lúc chạy, terminal in heartbeat mỗi 30 giây. Sau mỗi epoch còn có
`TrainLoss`, `ValLoss`, thời gian epoch và ETA.

Để rời tmux nhưng giữ job chạy, nhấn `Ctrl+B`, thả tay, rồi nhấn `D`.

Để quay lại xem:

```bash
tmux attach -t mednext_smoke
```

Chỉ được coi là pass khi cuối terminal có:

```text
MEDNEXT PIPELINE COMPLETED AND VERIFIED
```

Ghi lại `Experiment ID` và `Final report bundle` được in ngay bên dưới.

Nếu pipeline dừng vì mất kết nối/lỗi tạm thời, nó sẽ in chính xác lệnh resume.
Sao chép nguyên lệnh đó; không tự tạo experiment ID mới để resume.

## H. Kiểm tra smoke report

Sau khi smoke hoàn tất, chạy:

```bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
Code/.venv-models/bin/python -m glioma_seg.backends.mednext.smoke_gate \
  --project-root "$PWD" verify
```

Kết quả đúng phải chứa:

```json
"valid": true
```

Có thể liệt kê report vừa tạo bằng:

```bash
ls -1dt Workspace/reports/mednext_s_k3_smoke_* | head -n 1
```

## I. Chạy full MedNeXt 100 epoch × 5 folds

Chỉ làm phần này sau khi bước H trả về `"valid": true`.

Tạo tmux mới:

```bash
tmux new -s mednext_fullcv
```

Trong tmux, chạy đúng một lệnh:

```bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
bash Code/scripts/run_mednext_cv_pipeline.sh --confirm-run
```

Pipeline sẽ train tuần tự fold 0, 1, 2, 3, 4; mỗi fold 100 epoch. Sau đó nó tự
assemble đủ 1.251 OOF predictions, giữ NPZ, tính hai bộ metrics, thống kê failure,
tạo hình đúng orientation, tổng hợp telemetry và sinh report cuối.

Để detach: `Ctrl+B`, sau đó `D`. Để xem lại:

```bash
tmux attach -t mednext_fullcv
```

Không chạy một pipeline GPU thứ hai trên cùng RTX 3090. Script có GPU lock và sẽ
từ chối nếu phát hiện CUDA process khác.

## J. Resume đúng cách sau khi bị dừng

Khi có lỗi hoặc server restart, pipeline in một lệnh dạng:

```bash
bash Code/scripts/run_mednext_cv_pipeline.sh \
  --experiment-id mednext_s_k3_fullcv_YYYYMMDD_HHMMSS_xxxxxx \
  --resume --confirm-run
```

Mở tmux, `cd` vào repository thật, rồi chạy chính xác lệnh được in. Pipeline sẽ
audit owner/config/checkpoint trước khi resume. Nó không ghi đè một checkpoint
không khớp và không đoán experiment cần tiếp tục.

## K. Report cuối nằm ở đâu

Khi thành công, terminal in đường dẫn chính xác. Dạng chung là:

```text
/mnt/d/Henry/_storage/Glioma_3D_Segmentation_data/Workspace/reports/<EXPERIMENT_ID>
```

File đọc đầu tiên:

```text
summary.md
```

Chỉ gửi/zip bundle sau khi có `report_manifest.json` với
`"is_final_baseline": true`. Smoke report luôn có giá trị `false` và không phải
kết quả baseline cuối.
