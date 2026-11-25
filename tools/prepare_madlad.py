#!/usr/bin/env python3
"""
Подготовка MADLAD-400 3B для приложения VNMOff Wolf БЕЗ тяжёлого экспорта.

Готовые квантованные (int8) ONNX-модели скачиваются с Hugging Face
(CXDuncan/madlad400-3b-mt-optimized-quantized-onnx: энкодер, декодер, декодер с кэшем; Apache-2.0),
два декодера склеиваются в один (как ждёт приложение: вход use_cache_branch), веса сохраняются
во внешние .data-файлы. Тяжёлого вычисления нет: хватает обычного GitHub-раннера (15 ГБ ОЗУ).

Что делает:
  1. spiece.model и config.json из google/madlad400-3b-mt -> madlad-vocab.tsv и madlad-meta.tsv.
  2. Скачивает encoder/decoder/decoder_with_past (около 5 ГБ).
  3. Склеивает декодеры (optimum.onnx.merge_decoders), пересохраняет модели как <имя>.onnx + <имя>.onnx.data.
  4. Проверяет имена входов/выходов и размеры файлов, пробует перевести фразу (результат только в лог).

Результат в каталоге OUT (имена такие, как ждёт ModelStore в приложении):
  madlad-vocab.tsv  madlad-meta.tsv
  madlad-encoder.int8.onnx  madlad-encoder.int8.onnx.data
  madlad-decoder.int8.onnx  madlad-decoder.int8.onnx.data

Нужно: python 3.11, ~15 ГБ диска, ~12 ГБ ОЗУ.
  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
  pip install "numpy<2" "optimum[onnxruntime]==1.23.3" transformers==4.46.3 onnx==1.16.2 onnxruntime==1.19.2 sentencepiece huggingface_hub

Запуск:  python tools/prepare_madlad.py out/
"""
import json
import os
import shutil
import sys

MODEL = "google/madlad400-3b-mt"                                  # словарь и config
ONNX_REPO = "CXDuncan/madlad400-3b-mt-optimized-quantized-onnx"   # готовые int8-модели
LANGS = ["ru", "en", "ka"]
GITHUB_LIMIT = 2 * 1024 ** 3 - 1  # один файл релиза GitHub < 2 ГиБ

out = sys.argv[1] if len(sys.argv) > 1 else "out"
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


def save_external(model, final):
    import onnx

    for stale in (final, final + ".data"):
        if os.path.exists(stale):
            os.remove(stale)
    onnx.save_model(model, final, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(final) + ".data", size_threshold=1024)


def prepare_models():
    import onnx
    from huggingface_hub import hf_hub_download
    from optimum.onnx import merge_decoders

    enc_src = hf_hub_download(ONNX_REPO, "encoder_model_quantized.onnx")
    dec_src = hf_hub_download(ONNX_REPO, "decoder_model_quantized.onnx")
    past_src = hf_hub_download(ONNX_REPO, "decoder_with_past_model_quantized.onnx")
    for f in (enc_src, dec_src, past_src):
        print("скачано:", os.path.basename(f), f"{os.path.getsize(f) / 1e6:.0f} МБ")

    enc = onnx.load(enc_src)
    final_enc = os.path.join(out, "madlad-encoder.int8.onnx")
    save_external(enc, final_enc)
    del enc

    print("склеиваю декодеры…")
    try:
        merged = merge_decoders(dec_src, past_src)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Не удалось склеить декодеры: {e}\n"
            "Тогда приложению нужна поддержка двух отдельных декодеров (сообщите об этой ошибке)."
        )
    final_dec = os.path.join(out, "madlad-decoder.int8.onnx")
    save_external(merged, final_dec)
    del merged
    return final_enc, final_dec


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


def selftest(enc_path, dec_path):
    """Жадный перевод одной фразы без кэша. Только для лога: сбой здесь не останавливает workflow."""
    try:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from sentencepiece import SentencePieceProcessor

        sp = SentencePieceProcessor(model_file=hf_hub_download(MODEL, "spiece.model"))
        cfg = json.load(open(hf_hub_download(MODEL, "config.json"), encoding="utf-8"))
        eos, start = sp.piece_to_id("</s>"), cfg.get("decoder_start_token_id", 0)
        src = np.array([sp.encode("<2ru> Hello, how are you?") + [eos]], dtype=np.int64)
        mask = np.ones_like(src)

        opts = ort.SessionOptions()
        enc = ort.InferenceSession(enc_path, opts, providers=["CPUExecutionProvider"])
        hidden = enc.run(None, {"input_ids": src, "attention_mask": mask})[0]
        dec = ort.InferenceSession(dec_path, opts, providers=["CPUExecutionProvider"])
        pasts = {}
        for i in dec.get_inputs():
            if i.name.startswith("past_key_values."):
                heads, dim = i.shape[1], i.shape[3]
                pasts[i.name] = np.zeros((1, heads, 0, dim), dtype=np.float32)
        tokens = [start]
        for _ in range(40):
            feed = {"input_ids": np.array([tokens], dtype=np.int64), "encoder_hidden_states": hidden,
                    "encoder_attention_mask": mask, "use_cache_branch": np.array([False])}
            feed.update(pasts)
            names = [o.name for o in dec.get_outputs()]
            logits = dec.run(["logits"], feed)[0][0, -1]
            nxt = int(np.argmax(logits))
            if nxt == eos:
                break
            tokens.append(nxt)
        print("ТЕСТ ПЕРЕВОДА <2ru> Hello, how are you? ->", sp.decode(tokens[1:]))
    except Exception as e:  # noqa: BLE001
        print("тест перевода не выполнен (не критично):", repr(e))


if __name__ == "__main__":
    tokenizer_files()
    enc, dec = prepare_models()
    check(enc, False)
    check(dec, True)
    selftest(enc, dec)
    print("готово:", sorted(os.listdir(out)))
