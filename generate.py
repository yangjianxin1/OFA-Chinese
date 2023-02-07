from component.ofa.modeling_ofa import OFAModel
from torchvision import transforms
from PIL import Image
from glob import glob
from transformers import BertTokenizerFast


def main():
    model_name_or_path = 'YeungNLP/ofa-cn-base-muge-v2'
    image_path = './images/test/*'

    # 初始化model和tokenizer
    model = OFAModel.from_pretrained(model_name_or_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
    # 初始化图片预处理器
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    # 对prompt进行预处理
    prompt = '图片描述了什么?'
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids

    # 扫描目录下的所有图片
    for file in glob(image_path):
        # 加载图片并且进行预处理
        img = Image.open(file)
        patch_images = patch_resize_transform(img).unsqueeze(0)
        # 生成caption
        gen = model.generate(input_ids, patch_images=patch_images, num_beams=5, no_repeat_ngram_size=3)
        text = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].replace(' ', '')
        print(file)
        print(text)
        print()


if __name__ == '__main__':
    main()
