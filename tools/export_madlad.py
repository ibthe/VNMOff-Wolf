#!/usr/bin/env python3
"""
Экспорт google/madlad400-3b-mt в ONNX для приложения VNMOff Wolf.

Что делает:
  1. Скачивает spiece.model и config.json, готовит vocab.tsv и meta.tsv для токенизатора приложения.
  2. Экспортирует модель через optimum (энкодер + декодер с кэшем ключей/значений, merged).
  3. Квантует веса в uint8 (MatMul и таблица эмбеддингов) и сохраняет каждую модель как
     <имя>.onnx + <имя>.onnx.data (веса во внешнем файле: у протобуфа лимит 2 ГБ).

Результат в каталоге OUT (имена уже такие, как ждёт ModelStore в приложении):
  madlad-vocab.tsv  madlad-meta.tsv
  madlad-encoder.int8.onnx  madlad-encoder.int8.onnx.data
  madlad-decoder.int8.onnx  madlad-decoder.int8.onnx.data

Нужно: python 3.10+, ~35 ГБ свободного диска, ~32 ГБ ОЗУ (или большой своп).
  pip install torch==2.4.1 "numpy<2" "optimum[onnxruntime]==1.23.3" transformers==4.46.3 onnx==1.16.2 onnxruntime==1.19.2 sentencepiece huggingface_hub
  (на Python 3.11; с новыми версиями transformers/Python 3.14 экспорт optimum падает)

Запуск:  python tools/export_madlad.py out/
Если стандартного GitHub-раннера не хватает, запустите скрипт на своём компьютере/в Colab,
а получившиеся файлы выложите в релиз `models` вручную.
"""
import json
import os
import shutil
import subprocess
import sys

MODEL = "google/madlad400-3b-mt"
LANGS = ["ru", "en", "ka"]
GITHUB_LIMIT = 2 * 1024 ** 3 - 1  # один файл релиза GitHub < 2 ГиБ

out = sys.argv[1] if len(sys.argv) > 1 else "out"
raw = os.path.join(out, "_raw")
os.makedirs(out, exist_ok=True)


def read_spm_pieces(path):
    """Читает токены из spiece.model напрямую (формат protobuf), без пакета protobuf:
    его версия конфликтует с sentencepiece_model_pb2. Возвращает [(токен, оценка, тип)]."""
    import struct

    data = open(path, "rb").read()

    def varint(b, i):
        r = 0
        shift = 0
        while True:
            c = b[i]
            i += 1
            r |= (c & 0x7F) << shift
            shift += 7
            if not c & 0x80:
                return r, i

    def skip(b, i, wt):
        if wt == 0:
            return varint(b, i)[1]
        if wt == 1:
            return i + 8
        if wt == 5:
            return i + 4
        if wt == 2:
            ln, i = varint(b, i)
            return i + ln
        raise ValueError(f"неизвестный тип поля protobuf: {wt}")

    def parse_piece(b):
        piece, score, typ = "", 0.0, 1
        i = 0
        while i < len(b):
            key, i = varint(b, i)
            field, wt = key >> 3, key & 7
            if wt == 2 and field == 1:
                ln, i = varint(b, i)
                piece = b[i:i + ln].decode("utf-8")
                i += ln
            elif wt == 5 and field == 2:
                score = struct.unpack("<f", b[i:i + 4])[0]
                i += 4
            elif wt == 0 and field == 3:
                typ, i = varint(b, i)
            else:
                i = skip(b, i, wt)
        return piece, score, typ

    pieces = []
    i = 0
    while i < len(data):
        key, i = varint(data, i)
        field, wt = key >> 3, key & 7
        if wt == 2 and field == 1:
            ln, i = varint(data, i)
            pieces.append(parse_piece(data[i:i + ln]))
            i += ln
        else:
            i = skip(data, i, wt)
    return pieces


def tokenizer_files():
    from huggingface_hub import hf_hub_download
    from sentencepiece import SentencePieceProcessor

    spm_path = hf_hub_download(MODEL, "spiece.model")
    cfg = json.load(open(hf_hub_download(MODEL, "config.json"), encoding="utf-8"))
    sp = SentencePieceProcessor(model_file=spm_path)
    pieces = read_spm_pieces(spm_path)
    kind = {1: "n", 2: "u", 3: "c", 4: "d", 5: "x", 6: "b"}  # NORMAL UNKNOWN CONTROL USER_DEFINED UNUSED BYTE

    def clean(x):
        return x.replace("\t", " ").replace("\n", " ").replace("\r", " ")

    n = len(pieces)
    assert n == sp.get_piece_size(), f"разобрано {n} токенов, а в модели {sp.get_piece_size()}"
    for i in (0, 1, 2, 3, n // 2, n - 1):
        assert pieces[i][0] == sp.id_to_piece(i), f"разбор spiece.model не совпал на id {i}"
    with open(os.path.join(out, "madlad-vocab.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for piece, score, typ in pieces:
            f.write(f"{score}\t{clean(piece)}\t{kind.get(typ, 'n')}\n")

    # в spiece.model MADLAD нет токена «<pad>» (piece_to_id вернул бы id <unk>), поэтому id берём из config.json
    eos = sp.piece_to_id("</s>")
    unk = sp.unk_id()
    pad = cfg.get("pad_token_id", 0)
    start = cfg.get("decoder_start_token_id", 0)
    print("config:", {k: cfg.get(k) for k in ("eos_token_id", "pad_token_id", "decoder_start_token_id", "vocab_size")})
    print("spm   :", {"eos": eos, "pad": pad, "unk": unk, "pieces": n})
    assert eos == cfg.get("eos_token_id"), "id </s> в spiece.model и config.json не совпадают"
    with open(os.path.join(out, "madlad-meta.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"eos\t{eos}\npad\t{pad}\nunk\t{unk}\ndecoder_start\t{start}\n")
        for lang in LANGS:
            tag = f"<2{lang}>"
            tid = sp.piece_to_id(tag)
            assert tid != unk and sp.id_to_piece(tid) == tag, f"в словаре нет тега {tag}"
            f.write(f"{tag}\t{tid}\n")
    print("токенизатор готов")


def export_onnx():
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", MODEL,
        "--task", "text2text-generation-with-past",
        raw,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    for name in ("encoder_model.onnx", "decoder_model_merged.onnx"):
        assert os.path.exists(os.path.join(raw, name)), f"экспорт не создал {name}"


def quantize(src, dst_name):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp_dir = os.path.join(out, "_q")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, dst_name)
    quantize_dynamic(
        src, tmp,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gather"],
        use_external_data_format=True,
    )
    # пересохраняем с предсказуемым именем файла весов: рядом с .onnx лежит <имя>.data
    model = onnx.load(tmp)
    final = os.path.join(out, "madlad-" + dst_name)
    for stale in (final, final + ".data"):
        if os.path.exists(stale):
            os.remove(stale)
    onnx.save_model(model, final, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(final) + ".data", size_threshold=1024)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return final


def check(final, is_decoder):
    import onnx
    m = onnx.load(final, load_external_data=False)
    ins = [i.name for i in m.graph.input]
    outs = [o.name for o in m.graph.output]
    print(os.path.basename(final), "входы:", ins[:8], "... всего", len(ins), "| выходы:", outs[:4], "...")
    if is_decoder:
        need = {"input_ids", "encoder_hidden_states", "encoder_attention_mask", "use_cache_branch"}
        assert need <= set(ins), f"в декодере нет входов: {need - set(ins)}"
        assert "logits" in outs, "в декодере нет выхода logits"
        assert any(i.startswith("past_key_values.") and ".encoder." in i for i in ins), "нет past_key_values.*.encoder.*"
    else:
        assert {"input_ids", "attention_mask"} <= set(ins), "в энкодере нет input_ids/attention_mask"
    for f in (final, final + ".data"):
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f)}: {size / 1e6:.0f} МБ")
        assert size < GITHUB_LIMIT, f"{f} больше лимита GitHub (2 ГиБ)"


if __name__ == "__main__":
    tokenizer_files()
    export_onnx()
    enc = quantize(os.path.join(raw, "encoder_model.onnx"), "encoder.int8.onnx")
    check(enc, False)
    dec = quantize(os.path.join(raw, "decoder_model_merged.onnx"), "decoder.int8.onnx")
    check(dec, True)
    shutil.rmtree(raw, ignore_errors=True)
    print("готово:", sorted(os.listdir(out)))
