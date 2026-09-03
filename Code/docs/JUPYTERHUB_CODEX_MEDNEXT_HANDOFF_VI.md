# Handoff toàn diện cho Codex trên JupyterHub — MedNeXt

> Snapshot trạng thái: **2026-09-03 (Asia/Taipei)**  
> Repository: **Glioma 3D Segmentation**  
> Mục tiêu hiện tại: kiểm chứng rồi chạy baseline **MedNeXt-S-k3, 100 epochs ×
> 5 folds** trên RTX 3090, sau đó sinh bộ report tương đương baseline nnU-Net.

## 1. Cách dùng tài liệu này

Tài liệu này cung cấp bối cảnh để một phiên Codex mới trên JupyterHub không phải
đoán lại mục tiêu nghiên cứu, cấu trúc thư mục và các quyết định đã được khóa.
Nó **không thay thế việc đọc code**.

Thứ tự ưu tiên nguồn sự thật:

1. dữ liệu và artifact thực tế trên máy;
2. Git commit, submodule commit và source code đang checkout;
3. config/manifest/hash do pipeline ghi;
4. hai tài liệu handoff/runbook;
5. nội dung hội thoại hoặc giả định.

Nếu tài liệu và máy thực tế khác nhau, dừng ở thao tác read-only, chỉ rõ khác biệt
và không tự ý “sửa cho khớp”.

## 2. Bối cảnh nghiên cứu

### Dataset và bài toán

- Benchmark chính: **BraTS 2023 Adult Glioma Pre-Treatment**.
- Có **1.251 ca training có ground truth**.
- Mỗi ca phải có đúng 5 NIfTI: 4 MRI modalities và 1 segmentation.
- Bốn modalities: T1n, T1c, T2w và T2-FLAIR.
- Nhãn raw BraTS 2023: 0=background, 1=NCR/non-enhancing, 2=ED,
  3=ET. Không được giả định schema cũ có nhãn 4.
- Vùng đánh giá:
  - ET = {3}
  - TC = {1, 3}
  - WT = {1, 2, 3}
  - luôn kiểm tra quan hệ lồng nhau ET ⊆ TC ⊆ WT.

Không thêm đăng ký ảnh, skull stripping, N4, CLAHE, min-max normalization hoặc
histogram matching tự chế. Baseline phải dùng preprocessing thuộc framework đã
chọn và được lưu provenance đầy đủ.

### Mục tiêu so sánh

Ba baseline cần có protocol và artifact tương đương:

1. nnU-Net v2 3D full-resolution;
2. MedNeXt-S-k3;
3. SegResNet/MONAI.

Sau khi từng baseline hoàn tất độc lập mới thực hiện so sánh theo ET/TC/WT,
chọn model/checkpoint theo vùng hoặc ensemble. Baseline hiện tại không dùng
external data, synthetic data, GAN, pretrained weights, ensemble hoặc
post-processing.

### Trạng thái các baseline

- nnU-Net v2 3D full-resolution đã hoàn tất 100 epochs × 5 folds, 1.251 OOF
  cases, semantic metrics, official lesion-wise metrics, failure statistics,
  telemetry và report. Report lịch sử nằm dưới
  Workspace/reports/nnunetv2_3dfullres_fold0_fullcv_20260829_144741_795997_3e6003.
- SegResNet chạy ở local RTX 2080 Ti và đã được người dùng tạm dừng khoảng Fold
  0 Epoch 59 vì rất chậm. Không đụng, di chuyển hoặc khởi động lại experiment
  này khi đang làm MedNeXt, trừ khi người dùng yêu cầu riêng.
- MedNeXt Linux pipeline đã được viết và kiểm tra tĩnh/unit test ở local, nhưng
  **chưa có bằng chứng end-to-end 3 epochs trên RTX 3090**. Vì vậy tuyệt đối chưa
  được coi là sẵn sàng full training cho đến khi smoke gate pass trên server.

## 3. Máy JupyterHub và cấu trúc lưu trữ

### Thông tin đã quan sát

- Codex CLI: 0.153.0, model đang chọn gpt-5.6-sol.
- Python hệ thống JupyterHub: /opt/jupyterhub/bin/python, Python 3.10.12.
- GPU: NVIDIA GeForce RTX 3090, 24.576 MiB VRAM.
- RAM: khoảng 23 GiB; swap 6 GiB.
- Linux root /dev/sdd: khoảng 953 GiB trống tại lần kiểm tra.
- /mnt/d: vùng lưu trữ lớn, bền hơn cho dataset và Workspace.

Driver báo CUDA 13.1 chỉ cho biết giới hạn CUDA mà driver hỗ trợ. Môi trường dự
án dùng PyTorch 2.5.1+cu121; driver mới có thể chạy binary CUDA 12.1.

Codex báo không tìm thấy bubblewrap hệ thống nhưng đang dùng bản bundled. Đây
chưa phải lỗi của pipeline. Không dùng sudo để cài. Chỉ nhờ quản trị viên nếu
Codex thực sự không chạy được command do sandbox, không phải chỉ vì warning này.

### Đường dẫn thật và symbolic link

Repository Git phải nằm trên Linux filesystem vì /mnt/d không hỗ trợ thao tác
chmod mà Git cần cho metadata:

    REPO=/var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation

Alias để nhìn thấy code trong giao diện JupyterLab:

    /mnt/d/Henry/_storage/Glioma_3D_Segmentation_code
        -> /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation

Data và output bền vững:

    STORE=/mnt/d/Henry/_storage/Glioma_3D_Segmentation_data
    REPO/Datasets  -> STORE/Datasets
    REPO/Workspace -> STORE/Workspace

Dataset raw đúng phải ở:

    /mnt/d/Henry/_storage/Glioma_3D_Segmentation_data/Datasets/
    ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData

Luôn cd vào đường dẫn thật REPO để chạy Git/setup/pipeline. Alias ở /mnt/d chỉ
giúp mở file trong giao diện; không tạo bản copy thứ hai.

/var/tmp có thể bị quản trị viên/hệ thống dọn. Vì vậy code phải được commit và
push lên GitHub; dataset, Workspace và report phải ở /mnt/d. Virtualenv có thể
được tạo lại từ script nếu repo /var/tmp bị mất.

## 4. Trạng thái upload dataset tại thời điểm handoff

Lần quan sát gần nhất:

    Cases: 947
    NIfTI: 4733
    Size:  khoảng 11G

Mục tiêu bắt buộc:

    Cases: 1251
    NIfTI files: 6255

Do đó dataset **chưa hoàn tất** tại thời điểm snapshot. Không setup, preprocess,
smoke hoặc full train khi hai số chưa đúng.

Thông báo:

    du: cannot access '.../Untitled Folder': No such file or directory

nhiều khả năng là race trong lúc upload: JupyterLab tạo/đổi tên/xóa thư mục tạm
đúng lúc du đang duyệt. Nó không tự chứng minh dữ liệu hỏng, và không có bằng
chứng rằng lệnh watch tự nối Untitled Folder. Nếu trình duyệt bị reload thì
những file đã ghi xong vẫn còn, còn file đang gửi có thể bị hủy.

Xử lý an toàn: upload lại **toàn bộ thư mục nguồn** vào cùng thư mục đích, chọn
merge/overwrite file trùng nếu giao diện hỏi; không xóa partial dataset. Sau khi
giao diện báo xong, kiểm tra:

~~~bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
TRAIN="$PWD/Datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
test -d "$TRAIN" && echo "Dataset folder: EXISTS" || echo "Dataset folder: MISSING"
printf 'Cases: '
find "$TRAIN" -mindepth 1 -maxdepth 1 -type d -name 'BraTS-GLI-*' | wc -l
printf 'NIfTI files: '
find "$TRAIN" -type f -name '*.nii.gz' | wc -l
du -sh "$TRAIN" 2>/dev/null || true
~~~

Hai phép đếm chỉ là inventory sơ bộ. Validation nội dung 1.251 ca của pipeline
mới là cổng quyết định trước training.

## 5. Git và upstream provenance

Commit local chứa Linux MedNeXt pipeline tại thời điểm snapshot:

    bdc09541 Add fail-closed Linux MedNeXt pipeline

Trước khi làm việc trên server, local phải commit/push toàn bộ thay đổi mới và
server phải git pull --ff-only origin main. Không edit đồng thời local và server
vì sẽ tạo hai nguồn sự thật.

Codex đọc AGENTS.md khi khởi tạo một phiên làm việc. Vì phiên Codex hiện tại đã
được mở trước khi file handoff này xuất hiện trên server, sau khi pull hãy thoát
phiên đó, cd vào repository thật và mở một phiên Codex mới. Nếu tiếp tục phiên
cũ, phải yêu cầu nó đọc các file bằng prompt ở Mục 16; cách chắc chắn hơn vẫn là
mở phiên mới để project instructions được nạp từ đầu.

Các submodule được pin:

    External/BraTS-2023-Metrics  43c905242b2eecf421d4ab2da7af8ece9777d322
    External/MONAI               46a5272196a6c2590ca2589029eed8e4d56ff008
    External/MONAI-model-zoo     b9e4d04bb2a073110bde9e5c05c9690241e938b6
    External/MedNeXt             0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba
    External/nnUNet              0e495086eb108ff79afe106291e8c15bd2f2bc3a

Không sửa source upstream trong External/. Adapter, orchestration, evaluator,
report và mọi experiment tùy chỉnh phải nằm trong Code/. Untracked cache ở
submodule không được git add và không phải lý do để sửa upstream.

Kiểm tra Git an toàn trên server:

~~~bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
git status --short
git log -1 --oneline
git submodule status
git -C External/MedNeXt status --short
git -C External/MedNeXt rev-parse HEAD
~~~

## 6. Hợp đồng MedNeXt đã khóa

Source chuẩn: official MIC-DKFZ/MedNeXt, MedNeXt v1 dựa trên nnU-Net v1.

    repository: https://github.com/MIC-DKFZ/MedNeXt.git
    commit: 0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba
    package/version: mednextv1 1.7.0
    trainer: nnUNetTrainerV2_MedNeXt_S_kernel3
    planner: ExperimentPlanner3D_v21_customTargetSpacing_1x1x1
    plans: nnUNetPlansv2.1_trgSp_1x1x1
    data identifier: nnUNetData_plans_v2.1_trgSp_1x1x1
    full task: Task501_BraTS2023GLI
    smoke task: Task951_BraTS2023GLISmoke

Kiến trúc:

    variant: MedNeXt-S
    kernel: 3×3×3
    input channels: 4
    classes including background: 4
    base channels: 32
    expansion ratio: 2
    block counts: [2,2,2,2,2,2,2,2,2]
    deep supervision: true
    residual blocks: true
    residual up/down: true

Planning/training:

    target spacing: 1×1×1 mm
    patch: 128×128×128
    optimizer: AdamW
    initial LR: 0.001
    epsilon: 0.0001
    weight decay: 0.00003
    schedule: polynomial power 0.9 over requested duration
    train batches/epoch: 250
    validation batches/epoch: 50
    AMP: enabled
    deterministic: false (official behavior)
    checkpoint: every completed epoch

Official recipe gốc là 1.000 epochs. Thí nghiệm này cố ý dùng **100 epochs mỗi
fold** như một compute-limited comparison; phải ghi rõ và không gọi nó là full
official 1.000-epoch recipe.

### Batch size

Không hard-code batch_size=1 hoặc batch_size=2. Batch size phải đến từ output
của **official planner** cho patch 128³ trên môi trường thực tế. GPU memory
preflight chạy một forward + official loss + backward + optimizer step bằng dữ
liệu thật để quyết định cấu hình planner có fit RTX 3090 hay không.

Nếu preflight OOM, không tự ý giảm patch, đổi architecture, bật gradient
checkpointing, accumulation hay sửa batch. Ghi lại bằng chứng và hỏi người dùng
trước vì đó là một experiment/hardware adaptation mới.

### Inference và probabilities

    checkpoint: final
    sliding-window step: 0.5
    Gaussian weighting: on
    TTA: off
    post-processing: off
    native softmax: retain

Native probability channels:

    [background, NCR, ED, ET]

Canonical channels dùng để so sánh/ensemble:

    ET = p(ET)
    TC = p(NCR) + p(ET)
    WT = p(NCR) + p(ED) + p(ET)
    order = [ET, TC, WT]

## 7. Canonical five-fold split

Mọi model phải dùng chính xác split của nnU-Net baseline:

    case count: 1251
    folds: 5
    validation counts: [251, 250, 250, 250, 250]
    generator: sorted case IDs + sklearn KFold
    shuffle: true
    random_state: 12345
    splits_final.json SHA-256:
    a9b8aaef82974d52aa3652624c4902d0515c73a573f8bb8f24ad7982b943ed7b

Mỗi case phải xuất hiện đúng một lần trong OOF validation. Không tạo split mới,
không đổi seed và không dùng split nội bộ khác của model.

## 8. Bản đồ code cần đọc

### Entry points

- Code/scripts/setup_research_models_env.sh: tạo hai môi trường Python user
  level, pin dependency, cài MedNeXt/project editable và kiểm tra CUDA.
- Code/scripts/run_mednext_cv_pipeline.sh: orchestration Linux một lệnh,
  fail-closed, GPU lock, resume và 12 stage.
- Code/docs/JUPYTERHUB_MEDNEXT_RUNBOOK_VI.md: hướng dẫn thao tác cho người dùng.

### Config

- Code/configs/models/mednext.yaml: hợp đồng model/source/training/inference.
- Code/configs/experiments/mednext_100epoch_cv.yaml: 5 folds, 100 epochs và
  smoke 8 train + 2 validation cases.

### MedNeXt backend

- Code/src/glioma_seg/backends/mednext/config.py: validation strict của recipe;
  thay đổi không được review sẽ fail.
- dataset.py: inventory, canonical split và adapter nnU-Net v2 → v1.
- trainer.py: official preprocessing/trainer, checkpoint owner/config audit,
  memory preflight, training/resume/inference và heartbeat GPU.
- backend.py: CLI và lifecycle experiment, assemble OOF, probability export.
- smoke_gate.py: issue/verify gate gắn với code, config, split, GPU và report.
- compat.py: compatibility layer tối thiểu; không được biến thành fork model.

### Thành phần dùng chung

- Code/src/glioma_seg/data/validate.py: raw dataset validation.
- Code/src/glioma_seg/data/nnunet_conversion.py: common converted Dataset501.
- Code/src/glioma_seg/data/canonical_splits.py: tái tạo và hash exact split.
- Code/src/glioma_seg/evaluation/model_crossval.py: semantic OOF evaluation.
- Code/src/glioma_seg/evaluation/official_runner.py: official lesion-wise run.
- Code/src/glioma_seg/analysis/failure_statistics.py: definitions và tables.
- Code/src/glioma_seg/analysis/failure_analysis.py: ranking failure cases.
- Code/src/glioma_seg/visualization/: figures với orientation đã kiểm soát.
- Code/src/glioma_seg/reporting/model_runtime.py: fold evidence/telemetry.
- Code/src/glioma_seg/reporting/model_bundle.py: summary/report-bundle audit.
- Code/src/glioma_seg/ensembles/canonical_probabilities.py: native → canonical
  probability schema; chưa chạy ensemble ở baseline.

### Tests quan trọng

- Code/tests/test_mednext_backend.py
- Code/tests/test_linux_mednext_pipeline.py
- Các test chung evaluation/reporting/failure/ensemble.

## 9. Pipeline 12 stage làm gì

Trước Stage 1, bootstrap sẽ validate đủ 1.251 raw cases, convert sang Dataset501
dùng chung và tạo/verify exact canonical split/hash.

1. Khởi tạo experiment có owner manifest.
2. Kiểm tra environment, pinned upstream source và GPU.
3. Chạy test gate tương thích MedNeXt.
4. Copy/publish full raw-data validation evidence.
5. Official preprocessing và real-data GPU memory preflight.
6. Train tuần tự từng fold, audit sâu và resume an toàn.
7. Assemble OOF và tính semantic Dice/HD95.
8. Chạy official BraTS lesion-wise Dice/HD95 đã pin.
9. Tính failure statistics backend-neutral.
10. Rank failure và tạo figure đúng orientation.
11. Aggregate training/GPU/inference telemetry.
12. Sinh summary.md, weekly_discussion.md và audit report bundle.

Smoke dùng 8 train + 2 validation cases, Fold 0, 3 epochs. Nó cố ý dừng sau
Epoch 1, audit checkpoint rồi resume đến Epoch 3, sau đó vẫn chạy inference,
probability export, cả hai evaluator, failure analysis và report audit.

## 10. Failure-analysis contract dùng chung

Áp dụng riêng ET/TC/WT và dùng 3D 6-connectivity:

- complete false positive;
- complete false negative;
- under-segmentation: cả hai mask có mặt và FN voxels / GT voxels >= 0.25;
- over-segmentation: cả hai mask có mặt và
  FP voxels / predicted voxels >= 0.25;
- mixed error: đồng thời under và over;
- large semantic HD95: finite HD95 > 10 mm;
- fragmentation: số component prediction ít nhất bằng GT + 2;
- isolated FP component: component prediction >= 10 mm³ không overlap GT;
- remote FP component: isolated component hợp lệ cách GT ít nhất 10 mm;
- any major error: union các flag độc lập.

Các hàng lỗi có thể overlap, không cộng tỷ lệ để suy ra tổng. Không thay threshold
giữa các model.

## 11. Artifact bắt buộc

Report cuối phải có provenance và audit được, bao gồm tối thiểu:

- experiment/config/environment/data-validation/preprocessing manifests;
- fold manifest, checkpoint audit, runtime và GPU samples/summary cho từng fold;
- OOF masks đủ 1.251 ca;
- native softmax NPZ và canonical ET/TC/WT NPZ;
- per-case/per-fold/aggregate semantic Dice và HD95;
- official lesion-wise per-case và summary metrics + evaluator status;
- cross-validation integrity/artifact manifest;
- failure statistics, rankings, selected case list và figures đúng orientation;
- inference timing;
- summary.md, weekly_discussion.md, logs và final report manifest.

NPZ không phải file phụ có thể bỏ: giữ chúng cho so sánh vùng và ensemble sau.

## 12. Environment và lệnh chuẩn

Không cần source .../activate; script gọi thẳng interpreter đúng môi trường.

Sau khi dataset đủ và Git clean/pinned:

~~~bash
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
bash Code/scripts/setup_research_models_env.sh --python /opt/jupyterhub/bin/python
~~~

Script tạo:

    Code/.venv-models        Python 3.10, model/project/PyTorch CUDA
    Code/.venv-brats-metrics Python 3.9, official BraTS evaluator

Không dùng sudo. Nếu thiếu python3.9, conda và micromamba, báo blocker cho
người dùng/admin; không thay evaluator hoặc phiên bản một cách im lặng.

Smoke bắt buộc:

~~~bash
tmux new -s mednext_smoke
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
bash Code/scripts/run_mednext_cv_pipeline.sh --smoke-test --confirm-run
~~~

Chỉ pass khi thấy:

    MEDNEXT PIPELINE COMPLETED AND VERIFIED

và lệnh sau trả "valid": true:

~~~bash
Code/.venv-models/bin/python -m glioma_seg.backends.mednext.smoke_gate \
  --project-root "$PWD" verify
~~~

Full chỉ sau smoke pass:

~~~bash
tmux new -s mednext_fullcv
cd /var/tmp/henry-glioma-20260903/Glioma_3D_Segmentation
bash Code/scripts/run_mednext_cv_pipeline.sh --confirm-run
~~~

Resume chỉ bằng exact ID do run cũ in:

~~~bash
bash Code/scripts/run_mednext_cv_pipeline.sh \
  --experiment-id EXACT_ID --resume --confirm-run
~~~

Thêm --smoke-test nếu experiment đó là smoke. Không tự đoán ID.

## 13. Bằng chứng kiểm tra đã có và phần chưa kiểm chứng

Đã kiểm tra ở local cho commit Linux pipeline:

- full project test suite: 139 passed;
- research-model environment subset: 136 passed;
- Ruff: pass;
- mypy cho source thay đổi: pass;
- Bash syntax: pass;
- canonical split hash và fold counts: khớp chính xác.

Chưa kiểm chứng trên JupyterHub/RTX 3090:

- setup script hoàn tất trên máy server;
- official MedNeXt preprocessing thật;
- planner batch size thực tế;
- memory preflight 128³;
- forced-stop/resume checkpoint;
- end-to-end smoke inference/evaluation/report;
- thời gian full 5-fold.

Không nói “chắc chắn không lỗi”. Cách đúng để tăng độ tin cậy là audit → test →
preflight → smoke → verify gate → full.

## 14. Quy trình đầu tiên Codex trên server phải làm

Codex phải thực hiện từng nhóm dưới đây, báo kết quả rồi mới tiến tiếp:

1. Đọc hoàn toàn AGENTS.md, tài liệu này, runbook và hai YAML config.
2. Chỉ đọc trạng thái Git/submodule/symlink/GPU; chưa sửa và chưa train.
3. Xác nhận upload đã dừng hay đang chạy. Nếu chưa đủ 1251/6255, hướng dẫn
   người dùng hoàn tất upload; không xóa partial dataset.
4. Sau khi đủ inventory, chạy project raw-data validation; mọi ca phải pass.
5. Xác nhận repo chứa commit mới nhất đã push từ local và worktree không có thay
   đổi người dùng chưa hiểu.
6. Review Linux setup/runner và chạy narrow tests. Chỉ sửa code project-owned
   trong Code/ nếu có lỗi thật, kèm test hồi quy; không sửa External/.
7. Chạy setup environment không sudo.
8. Kiểm tra nvidia-smi không có compute job khác.
9. Chạy memory preflight/smoke trong tmux, theo dõi heartbeat, artifact và
   checkpoint-resume.
10. Verify smoke gate. Chỉ khi valid mới đưa/chạy lệnh full theo yêu cầu.

Nếu full run gặp lỗi/restart: bảo toàn artifact, đọc log, audit fold và resume
exact experiment. Không tạo experiment mới chỉ để né lỗi và không ghi đè evidence.

## 15. Cách Codex nên trao đổi với người dùng

Người dùng muốn hướng dẫn rất cụ thể, dễ làm theo. Mỗi lần:

1. nói trạng thái hiện tại bằng một câu (đang upload, chưa đủ, pass,
   warning không fatal, đã dừng vì ...);
2. đưa một block lệnh copy/paste;
3. giải thích block đó chỉ đọc hay có thay đổi gì;
4. ghi kết quả mong đợi;
5. chờ output thực tế trước khi đưa bước kế tiếp nếu bước sau phụ thuộc kết quả.

Không suy luận “train đang ổn” chỉ vì GPU 100%; cần heartbeat/log/checkpoint. Không
gọi warning là error fatal. Không cam kết thời gian full từ một epoch đầu tiên;
dùng nhiều epoch ổn định và tách ETA của một fold với ETA toàn pipeline.

## 16. Prompt khởi động đề xuất cho phiên Codex JupyterHub

Sau khi local đã commit/push tài liệu này và server đã pull, có thể gửi cho Codex:

~~~text
Hãy đọc toàn bộ AGENTS.md,
Code/docs/JUPYTERHUB_CODEX_MEDNEXT_HANDOFF_VI.md,
Code/docs/JUPYTERHUB_MEDNEXT_RUNBOOK_VI.md,
Code/configs/models/mednext.yaml và
Code/configs/experiments/mednext_100epoch_cv.yaml. Sau đó chỉ audit read-only
trạng thái Git, submodules, symbolic links, dataset upload và GPU; chưa setup,
chưa sửa code, chưa preprocess và chưa train. Báo rõ phần nào đã pass, phần nào
chưa xác minh, rồi đưa cho tôi đúng một block lệnh an toàn tiếp theo. Tuyệt đối
không sửa External/, không dùng sudo, không xóa dữ liệu/Workspace, và không chạy
full MedNeXt trước khi complete 3-epoch smoke gate valid trên RTX 3090.
~~~
