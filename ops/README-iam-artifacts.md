# IAM: granting write access to the ML-artifacts bucket

The app credentials (`pythia-app`) are **read-only** on `s3:::pythia-ml-artifacts`
(`s3:PutObject` denied), which is why the sports artifacts could never be
uploaded and the S3 auto-download path stayed empty. `iam-artifact-uploader-policy.json`
grants exactly the permissions needed to populate the bucket — nothing more.

## What the policy allows
- `ListBucket` / `ListBucketMultipartUploads` on the bucket itself.
- `GetObject` / `PutObject` / `AbortMultipartUpload` / `ListMultipartUploadParts`
  on `artifacts/*` only (the prefix the deploy reads).
- The multipart actions matter: the 302MB `mlb_baseline` model uploads as a
  multipart object — that was the exact call (`CreateMultipartUpload`) that the
  read-only user was denied.

## Apply it (AWS Console)
Two options — attach to the existing user, or (cleaner) make a dedicated uploader.

**Option A — dedicated uploader user (recommended, keeps app read-only):**
1. IAM → Policies → Create policy → JSON tab → paste `iam-artifact-uploader-policy.json` → name it `pythia-artifact-uploader`.
2. IAM → Users → Create user `pythia-artifact-uploader` → attach `pythia-artifact-uploader` policy.
3. Create an access key for it (Application running outside AWS).
4. Use those keys ONLY for the one-time sync below — do not put them in the app `.env`.

**Option B — grant the existing app user write (simpler, broader):**
1. IAM → Users → `pythia-app` → Add permissions → attach `pythia-artifact-uploader` policy.
   (Leaves the app able to write artifacts; fine if acceptable.)

## Apply it (AWS CLI, if you have an admin profile)
```bash
aws iam create-policy --policy-name pythia-artifact-uploader \
  --policy-document file://ops/iam-artifact-uploader-policy.json
# then attach to a user:
aws iam attach-user-policy --user-name pythia-artifact-uploader \
  --policy-arn arn:aws:iam::874119032317:policy/pythia-artifact-uploader
```

## Then finish the self-healing (from EC2 host)
```bash
cd /home/ec2-user/seekingbeta
# export the write-capable keys for this shell only:
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
aws s3 sync pythia_divination/artifacts/ s3://pythia-ml-artifacts/artifacts/artifacts/ --only-show-errors
# enable auto-download so fresh boxes self-restore on start:
grep -q AUTO_DOWNLOAD_ARTIFACTS .env || echo 'AUTO_DOWNLOAD_ARTIFACTS=true' >> .env
docker compose up -d --force-recreate divination-api
# verify:
docker compose logs --tail=30 divination-api | grep -i artifact
```

After the sync, a fresh box/EBS restores all artifacts automatically on boot —
closing the last durability gap. See TENNIS_AND_TLS_RUNBOOK.md §3 for context.
