package com.example.voicenotes.translate

import java.io.ByteArrayOutputStream
import java.io.File
import java.text.Normalizer

/**
 * Токенизатор MADLAD-400: SentencePiece Unigram (лучшая разбивка алгоритмом Витерби).
 *
 * Читает два файла, которые делает workflow «Publish models» (tools/export_madlad.py) из spiece.model:
 *  - vocab.tsv: строка N = токен с id N, формат «оценка<TAB>токен<TAB>тип»,
 *    тип: n обычный, b байтовый (<0xNN>), c служебный, u неизвестный, d пользовательский (<2ru>…), x неиспользуемый;
 *  - meta.tsv: «имя<TAB>id»: eos, pad, unk, decoder_start и теги целевого языка (<2ru>, <2en>, <2ka>…).
 *
 * MADLAD не использует код исходного языка: в начало входа ставится тег ЦЕЛЕВОГО языка («<2en> Привет»),
 * в конец — </s>. Нормализация упрощённая (NFKC + сжатие пробелов).
 */
class MadladTokenizer(dir: File) {
    private val pieceToId = HashMap<String, Int>(300_000)
    private val scores: FloatArray
    private val types: ByteArray
    private val meta = HashMap<String, Int>()
    private val bytePiece = IntArray(256) { -1 }
    private var hasBytes = false
    private var maxPieceLen = 1
    private val unkScore: Float

    val eosId: Int
    val padId: Int
    val unkId: Int
    /** С какого токена декодер начинает перевод (decoder_start_token_id модели). */
    val decoderStartId: Int
    /** Число токенов в словаре SentencePiece; логиты за этим пределом (выравнивание таблицы) игнорируются. */
    val vocabSize: Int
    /** Токены, которые никогда не должны попадать в перевод (служебные, теги языков), кроме </s>. */
    val blockedIds: IntArray

    init {
        val sc = ArrayList<Float>(260_000)
        val ty = ArrayList<Byte>(260_000)
        var minScore = 0f
        File(dir, "vocab.tsv").bufferedReader(Charsets.UTF_8).useLines { lines ->
            var id = 0
            for (line in lines) {
                val p = line.split('\t', limit = 3)
                val score = p.getOrNull(0)?.toFloatOrNull() ?: 0f
                val piece = p.getOrNull(1) ?: ""
                val type = p.getOrNull(2)?.firstOrNull() ?: 'n'
                sc.add(score)
                ty.add(type.code.toByte())
                when (type) {
                    'n' -> if (piece.isNotEmpty()) {
                        pieceToId.putIfAbsent(piece, id)
                        if (piece.length > maxPieceLen) maxPieceLen = piece.length
                        if (score < minScore) minScore = score
                    }
                    'b' -> if (piece.length == 6 && piece.startsWith("<0x") && piece.endsWith(">")) {
                        val b = piece.substring(3, 5).toIntOrNull(16)
                        if (b != null) { bytePiece[b] = id; hasBytes = true }
                    }
                }
                id++
            }
        }
        scores = sc.toFloatArray()
        types = ty.toByteArray()
        unkScore = minScore - 10f

        File(dir, "meta.tsv").bufferedReader(Charsets.UTF_8).useLines { lines ->
            for (line in lines) {
                val tab = line.indexOf('\t')
                if (tab <= 0) continue
                val v = line.substring(tab + 1).trim().toIntOrNull() ?: continue
                meta[line.substring(0, tab)] = v
            }
        }
        fun need(k: String) = meta[k] ?: error("В meta.tsv нет «$k»")
        eosId = need("eos")
        padId = need("pad")
        unkId = need("unk")
        decoderStartId = need("decoder_start")
        vocabSize = scores.size
        blockedIds = (0 until scores.size).filter {
            it != eosId && types[it] != 'n'.code.toByte() && types[it] != 'b'.code.toByte()
        }.toIntArray()
    }

    /** id тега целевого языка: «ru» → <2ru>. */
    private fun targetTagId(lang: String): Int = meta["<2$lang>"] ?: error("В словаре нет языка $lang")

    /** Вход энкодера MADLAD: [<2целевой язык>] + токены + </s>. */
    fun encode(text: String, dstLang: String): IntArray {
        val out = ArrayList<Int>()
        out.add(targetTagId(dstLang))
        val norm = Normalizer.normalize(text, Normalizer.Form.NFKC)
            .replace(Regex("\\s+"), " ")
            .trim()
        if (norm.isNotEmpty()) {
            for (word in norm.split(' ')) {
                if (word.isNotEmpty()) encodeWord("\u2581$word", out)
            }
        }
        out.add(eosId)
        return out.toIntArray()
    }

    /** Символ, которого нет в словаре: байты UTF-8 (если в модели есть байтовые токены) или <unk>. */
    private fun addUnknown(ch: String, out: MutableList<Int>) {
        if (!hasBytes) { out.add(unkId); return }
        for (b in ch.toByteArray(Charsets.UTF_8)) {
            val id = bytePiece[b.toInt() and 0xFF]
            out.add(if (id >= 0) id else unkId)
        }
    }

    /** Лучшая разбивка одного слова (с ведущим «▁») по оценкам словаря. */
    private fun encodeWord(word: String, out: MutableList<Int>) {
        val n = word.length
        val best = FloatArray(n + 1) { Float.NEGATIVE_INFINITY }
        val from = IntArray(n + 1) { -1 }
        val tokenAt = IntArray(n + 1) { -1 }   // -1: символ уходит в байты / <unk>
        best[0] = 0f
        for (i in 0 until n) {
            if (best[i] == Float.NEGATIVE_INFINITY) continue
            // не режем суррогатные пары посередине
            val maxEnd = minOf(n, i + maxPieceLen)
            var end = i + 1
            var foundSingle = false
            while (end <= maxEnd) {
                if (end < n && Character.isLowSurrogate(word[end])) { end++; continue }
                val id = pieceToId[word.substring(i, end)]
                if (id != null) {
                    val s = best[i] + scores[id]
                    if (s > best[end]) { best[end] = s; from[end] = i; tokenAt[end] = id }
                    if (end == i + 1 || (end == i + 2 && Character.isHighSurrogate(word[i]))) foundSingle = true
                }
                end++
            }
            if (!foundSingle) {
                val e = if (Character.isHighSurrogate(word[i]) && i + 2 <= n) i + 2 else i + 1
                val s = best[i] + unkScore
                if (s > best[e]) { best[e] = s; from[e] = i; tokenAt[e] = -1 }
            }
        }
        if (best[n] == Float.NEGATIVE_INFINITY) { out.add(unkId); return }
        val spans = ArrayList<IntArray>()          // [начало, конец, id]
        var pos = n
        while (pos > 0) {
            spans.add(intArrayOf(from[pos], pos, tokenAt[pos]))
            pos = from[pos]
        }
        for (k in spans.indices.reversed()) {
            val (a, b, id) = spans[k]
            if (id >= 0) out.add(id) else addUnknown(word.substring(a, b), out)
        }
    }

    private val idToPiece: Array<String?> by lazy {
        val arr = arrayOfNulls<String>(scores.size)
        for ((p, id) in pieceToId) if (id < arr.size) arr[id] = p
        arr
    }

    /** Склеивает токены в текст; служебные пропускаются, байтовые токены собираются в UTF-8. */
    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        val bytes = ByteArrayOutputStream()
        fun flush() {
            if (bytes.size() > 0) { sb.append(String(bytes.toByteArray(), Charsets.UTF_8)); bytes.reset() }
        }
        val byteType = 'b'.code.toByte()
        for (id in ids) {
            if (id < 0 || id >= scores.size) continue
            if (types[id] == byteType) {
                val b = (0..255).firstOrNull { bytePiece[it] == id }
                if (b != null) bytes.write(b)
                continue
            }
            flush()
            val p = idToPiece[id] ?: continue
            sb.append(p)
        }
        flush()
        return sb.toString().replace('\u2581', ' ').trim()
    }
}
