#DANN
python3 -m model.main --config model/DANN/wmb1i1-inpainting/DANN.yaml
python3 -m model.main --config model/DANN/wmb1i1-mb0i0/DANN.yaml
python3 -m model.main --config model/DANN/wmb1i1-wmb2i2/DANN.yaml

python3 -m model.main --config model/DANN/mb0i0-inpainting/DANN.yaml
python3 -m model.main --config model/DANN/mb0i0-wmb1i1/DANN.yaml
python3 -m model.main --config model/DANN/mb0i0-wmb2i2/DANN.yaml

python3 -m model.main --config model/DANN/inpainting-mb0i0/DANN.yaml
python3 -m model.main --config model/DANN/inpainting-wmb1i1/DANN.yaml
python3 -m model.main --config model/DANN/inpainting-wmb2i2/DANN.yaml


#DSAN
python3 -m model.main --config model/DSAN/wmb1i1-inpainting/DSAN.yaml
python3 -m model.main --config model/DSAN/wmb1i1-mb0i0/DSAN.yaml
python3 -m model.main --config model/DSAN/wmb1i1-wmb2i2/DSAN.yaml

python3 -m model.main --config model/DSAN/mb0i0-inpainting/DSAN.yaml
python3 -m model.main --config model/DSAN/mb0i0-wmb1i1/DSAN.yaml
python3 -m model.main --config model/DSAN/mb0i0-wmb2i2/DSAN.yaml

python3 -m model.main --config model/DSAN/inpainting-mb0i0/DSAN.yaml
python3 -m model.main --config model/DSAN/inpainting-wmb1i1/DSAN.yaml
python3 -m model.main --config model/DSAN/inpainting-wmb2i2/DSAN.yaml

#GRAM
python3 -m model.main --config model/GRAM/wmb1i1-inpainting/GRAM.yaml
python3 -m model.main --config model/GRAM/wmb1i1-mb0i0/GRAM.yaml
python3 -m model.main --config model/GRAM/wmb1i1-wmb2i2/GRAM.yaml

python3 -m model.main --config model/GRAM/mb0i0-inpainting/GRAM.yaml
python3 -m model.main --config model/GRAM/mb0i0-wmb1i1/GRAM.yaml
python3 -m model.main --config model/GRAM/mb0i0-wmb2i2/GRAM.yaml

python3 -m model.main --config model/GRAM/inpainting-mb0i0/GRAM.yaml
python3 -m model.main --config model/GRAM/inpainting-wmb1i1/GRAM.yaml
python3 -m model.main --config model/GRAM/inpainting-wmb2i2/GRAM.yaml



