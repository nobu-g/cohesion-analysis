import re

# ---------- word module|word feature tagging ----------
SUB_WORD_FEATURES = ("用言表記先頭", "用言表記末尾")  # メンテナンスしない単語素性
WORD_FEATURES: tuple[str, ...] = ("基本句-主辞", "基本句-区切", "文節-区切", *SUB_WORD_FEATURES)

# ---------- word module|base phrase feature tagging ----------
SUB_BASE_PHRASE_FEATURES = (  # メンテナンスしない基本句素性
    # cf. https://github.com/ku-nlp/knp/blob/master/doc/knp_feature.pdf
    "SM-主体",
    "レベル:A",
    "レベル:A-",
    "レベル:B",
    "レベル:B+",
    "レベル:B-",
    "レベル:C",
    "係:ノ格",
    "修飾",
    "状態述語",
    "動態述語",
    "可能表現",
    "敬語:尊敬表現",
    "敬語:謙譲表現",
    "敬語:丁寧表現",
    "時間",
    "節-区切:補文",
    "節-区切:連体修飾",
    # cf. https://github.com/ku-nlp/KWDLC/blob/master/doc/clause_feature_manual.pdf
    "節-機能-原因・理由",
    "節-機能疑-原因・理由",
    "節-前向き機能-原因・理由",
    "節-前向き機能-原因・理由-逆",
    "節-機能-目的",
    "節-機能疑-目的",
    "節-前向き機能-目的",
    "節-機能-条件",
    "節-機能疑-条件",
    "節-前向き機能-条件",
    "節-前向き機能-否定条件",
    "節-前向き機能-対比",
    "節-機能-逆接",
    "節-機能疑-逆接",
    "節-前向き機能-逆接",
    "節-機能-条件-逆条件",
    "節-機能疑-条件-逆条件",
    "節-機能-逆接",
    "節-機能疑-逆接",
    "節-前向き機能-逆接",
    "節-機能-時間経過-前",
    "節-機能-時間経過-後",
    "節-機能-時間経過-同時",
)
BASE_PHRASE_FEATURES = (
    # type
    "用言:動",
    "用言:形",
    "用言:判",
    "体言",
    "非用言格解析:動",
    "非用言格解析:形",
    "非用言格解析:不明",
    # modality
    "モダリティ-疑問",
    "モダリティ-意志",
    "モダリティ-勧誘",
    "モダリティ-命令",
    "モダリティ-禁止",
    "モダリティ-評価:弱",
    "モダリティ-評価:強",
    "モダリティ-認識-推量",
    "モダリティ-認識-蓋然性",
    "モダリティ-認識-証拠性",
    "モダリティ-依頼Ａ",
    "モダリティ-依頼Ｂ",
    # tense
    "時制:過去",
    "時制:非過去",
    # negation
    "否定表現",
    # clause
    "節-主辞",
    "節-区切",
    *SUB_BASE_PHRASE_FEATURES,
)
IGNORE_VALUE_FEATURE_PAT = re.compile(r"節-(前向き)?機能疑?")


# ---------- word module|morphological analysis ----------
# 品詞
POS_TAG2POS_ID = {
    "特殊": 1,
    "動詞": 2,
    "形容詞": 3,
    "判定詞": 4,
    "助動詞": 5,
    "名詞": 6,
    "指示詞": 7,
    "副詞": 8,
    "助詞": 9,
    "接続詞": 10,
    "連体詞": 11,
    "感動詞": 12,
    "接頭辞": 13,
    "接尾辞": 14,
}

# 品詞細分類
POS_TAG_SUBPOS_TAG2SUBPOS_ID: dict[str, dict[str, int]] = {
    "特殊": {
        "句点": 1,
        "読点": 2,
        "括弧始": 3,
        "括弧終": 4,
        "記号": 5,
        "空白": 6,
    },
    "動詞": {"*": 0},
    "形容詞": {"*": 0},
    "判定詞": {"*": 0},
    "助動詞": {"*": 0},
    "名詞": {
        "普通名詞": 1,
        "サ変名詞": 2,
        "固有名詞": 3,
        "地名": 4,
        "人名": 5,
        "組織名": 6,
        "数詞": 7,
        "形式名詞": 8,
        "副詞的名詞": 9,
        "時相名詞": 10,
    },
    "指示詞": {
        "名詞形態指示詞": 1,
        "連体詞形態指示詞": 2,
        "副詞形態指示詞": 3,
    },
    "副詞": {"*": 0},
    "助詞": {
        "格助詞": 1,
        "副助詞": 2,
        "接続助詞": 3,
        "終助詞": 4,
    },
    "接続詞": {"*": 0},
    "連体詞": {"*": 0},
    "感動詞": {"*": 0},
    "接頭辞": {
        "名詞接頭辞": 1,
        "動詞接頭辞": 2,
        "イ形容詞接頭辞": 3,
        "ナ形容詞接頭辞": 4,
    },
    "接尾辞": {
        "名詞性述語接尾辞": 1,
        "名詞性名詞接尾辞": 2,
        "名詞性名詞助数辞": 3,
        "名詞性特殊接尾辞": 4,
        "形容詞性述語接尾辞": 5,
        "形容詞性名詞接尾辞": 6,
        "動詞性接尾辞": 7,
    },
}

# 活用型
CONJTYPE_TAGS = (
    "*",
    "母音動詞",
    "子音動詞カ行",
    "子音動詞カ行促音便形",
    "子音動詞ガ行",
    "子音動詞サ行",
    "子音動詞タ行",
    "子音動詞ナ行",
    "子音動詞バ行",
    "子音動詞マ行",
    "子音動詞ラ行",
    "子音動詞ラ行イ形",
    "子音動詞ワ行",
    "子音動詞ワ行文語音便形",
    "カ変動詞",
    "カ変動詞来",
    "サ変動詞",
    "ザ変動詞",
    "イ形容詞アウオ段",
    "イ形容詞イ段",
    "イ形容詞イ段特殊",
    "ナ形容詞",
    "ナノ形容詞",
    "ナ形容詞特殊",
    "タル形容詞",
    "判定詞",
    "無活用型",
    "助動詞ぬ型",
    "助動詞だろう型",
    "助動詞そうだ型",
    "助動詞く型",
    "動詞性接尾辞ます型",
    "動詞性接尾辞うる型",
)

# 活用形
CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID: dict[str, dict[str, int]] = {
    "*": {"*": 0},
    "母音動詞": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
        "文語命令形": 18,
    },
    "子音動詞カ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞カ行促音便形": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞ガ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞サ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞タ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞ナ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞バ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞マ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞ラ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞ラ行イ形": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "子音動詞ワ行": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
    },
    "子音動詞ワ行文語音便形": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
    },
    "カ変動詞": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "カ変動詞来": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
    },
    "サ変動詞": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
        "文語基本形": 18,
        "文語未然形": 19,
        "文語命令形": 20,
    },
    "ザ変動詞": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "基本条件形": 7,
        "基本連用形": 8,
        "タ接連用形": 9,
        "タ形": 10,
        "タ系推量形": 11,
        "タ系省略推量形": 12,
        "タ系条件形": 13,
        "タ系連用テ形": 14,
        "タ系連用タリ形": 15,
        "タ系連用チャ形": 16,
        "音便条件形": 17,
        "文語基本形": 18,
        "文語未然形": 19,
        "文語命令形": 20,
    },
    "イ形容詞アウオ段": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "命令形": 3,
        "基本推量形": 4,
        "基本省略推量形": 5,
        "基本条件形": 6,
        "基本連用形": 7,
        "タ形": 8,
        "タ系推量形": 9,
        "タ系省略推量形": 10,
        "タ系条件形": 11,
        "タ系連用テ形": 12,
        "タ系連用タリ形": 13,
        "タ系連用チャ形": 14,
        "タ系連用チャ形２": 15,
        "音便条件形": 16,
        "音便条件形２": 17,
        "文語基本形": 18,
        "文語未然形": 19,
        "文語連用形": 20,
        "文語連体形": 21,
        "文語命令形": 22,
        "エ基本形": 23,
    },
    "イ形容詞イ段": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "命令形": 3,
        "基本推量形": 4,
        "基本省略推量形": 5,
        "基本条件形": 6,
        "基本連用形": 7,
        "タ形": 8,
        "タ系推量形": 9,
        "タ系省略推量形": 10,
        "タ系条件形": 11,
        "タ系連用テ形": 12,
        "タ系連用タリ形": 13,
        "タ系連用チャ形": 14,
        "タ系連用チャ形２": 15,
        "音便条件形": 16,
        "音便条件形２": 17,
        "文語基本形": 18,
        "文語未然形": 19,
        "文語連用形": 20,
        "文語連体形": 21,
        "文語命令形": 22,
        "エ基本形": 23,
    },
    "イ形容詞イ段特殊": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "命令形": 3,
        "基本推量形": 4,
        "基本省略推量形": 5,
        "基本条件形": 6,
        "基本連用形": 7,
        "タ形": 8,
        "タ系推量形": 9,
        "タ系省略推量形": 10,
        "タ系条件形": 11,
        "タ系連用テ形": 12,
        "タ系連用タリ形": 13,
        "タ系連用チャ形": 14,
        "タ系連用チャ形２": 15,
        "音便条件形": 16,
        "音便条件形２": 17,
        "文語基本形": 18,
        "文語未然形": 19,
        "文語連用形": 20,
        "文語連体形": 21,
        "文語命令形": 22,
        "エ基本形": 23,
    },
    "ナ形容詞": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "ダ列基本連体形": 3,
        "ダ列基本推量形": 4,
        "ダ列基本省略推量形": 5,
        "ダ列基本条件形": 6,
        "ダ列基本連用形": 7,
        "ダ列タ形": 8,
        "ダ列タ系推量形": 9,
        "ダ列タ系省略推量形": 10,
        "ダ列タ系条件形": 11,
        "ダ列タ系連用テ形": 12,
        "ダ列タ系連用タリ形": 13,
        "ダ列タ系連用ジャ形": 14,
        "ダ列文語連体形": 15,
        "ダ列文語条件形": 16,
        "デアル列基本形": 17,
        "デアル列命令形": 18,
        "デアル列基本推量形": 19,
        "デアル列基本省略推量形": 20,
        "デアル列基本条件形": 21,
        "デアル列基本連用形": 22,
        "デアル列タ形": 23,
        "デアル列タ系推量形": 24,
        "デアル列タ系省略推量形": 25,
        "デアル列タ系条件形": 26,
        "デアル列タ系連用テ形": 27,
        "デアル列タ系連用タリ形": 28,
        "デス列基本形": 29,
        "デス列音便基本形": 30,
        "デス列基本推量形": 31,
        "デス列音便基本推量形": 32,
        "デス列基本省略推量形": 33,
        "デス列音便基本省略推量形": 34,
        "デス列タ形": 35,
        "デス列タ系推量形": 36,
        "デス列タ系省略推量形": 37,
        "デス列タ系条件形": 38,
        "デス列タ系連用テ形": 39,
        "デス列タ系連用タリ形": 40,
        "ヤ列基本形": 41,
        "ヤ列基本推量形": 42,
        "ヤ列基本省略推量形": 43,
        "ヤ列タ形": 44,
        "ヤ列タ系推量形": 45,
        "ヤ列タ系省略推量形": 46,
        "ヤ列タ系条件形": 47,
        "ヤ列タ系連用タリ形": 48,
    },
    "ナノ形容詞": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "ダ列基本連体形": 3,
        "ダ列特殊連体形": 4,
        "ダ列基本推量形": 5,
        "ダ列基本省略推量形": 6,
        "ダ列基本条件形": 7,
        "ダ列基本連用形": 8,
        "ダ列タ形": 9,
        "ダ列タ系推量形": 10,
        "ダ列タ系省略推量形": 11,
        "ダ列タ系条件形": 12,
        "ダ列タ系連用テ形": 13,
        "ダ列タ系連用タリ形": 14,
        "ダ列タ系連用ジャ形": 15,
        "ダ列文語連体形": 16,
        "ダ列文語条件形": 17,
        "デアル列基本形": 18,
        "デアル列命令形": 19,
        "デアル列基本推量形": 20,
        "デアル列基本省略推量形": 21,
        "デアル列基本条件形": 22,
        "デアル列基本連用形": 23,
        "デアル列タ形": 24,
        "デアル列タ系推量形": 25,
        "デアル列タ系省略推量形": 26,
        "デアル列タ系条件形": 27,
        "デアル列タ系連用テ形": 28,
        "デアル列タ系連用タリ形": 29,
        "デス列基本形": 30,
        "デス列音便基本形": 31,
        "デス列基本推量形": 32,
        "デス列音便基本推量形": 33,
        "デス列基本省略推量形": 34,
        "デス列音便基本省略推量形": 35,
        "デス列タ形": 36,
        "デス列タ系推量形": 37,
        "デス列タ系省略推量形": 38,
        "デス列タ系条件形": 39,
        "デス列タ系連用テ形": 40,
        "デス列タ系連用タリ形": 41,
        "ヤ列基本形": 42,
        "ヤ列基本推量形": 43,
        "ヤ列基本省略推量形": 44,
        "ヤ列タ形": 45,
        "ヤ列タ系推量形": 46,
        "ヤ列タ系省略推量形": 47,
        "ヤ列タ系条件形": 48,
        "ヤ列タ系連用タリ形": 49,
    },
    "ナ形容詞特殊": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "ダ列基本連体形": 3,
        "ダ列特殊連体形": 4,
        "ダ列基本推量形": 5,
        "ダ列基本省略推量形": 6,
        "ダ列基本条件形": 7,
        "ダ列基本連用形": 8,
        "ダ列特殊連用形": 9,
        "ダ列タ形": 10,
        "ダ列タ系推量形": 11,
        "ダ列タ系省略推量形": 12,
        "ダ列タ系条件形": 13,
        "ダ列タ系連用テ形": 14,
        "ダ列タ系連用タリ形": 15,
        "ダ列タ系連用ジャ形": 16,
        "ダ列文語連体形": 17,
        "ダ列文語条件形": 18,
        "デアル列基本形": 19,
        "デアル列命令形": 20,
        "デアル列基本推量形": 21,
        "デアル列基本省略推量形": 22,
        "デアル列基本条件形": 23,
        "デアル列基本連用形": 24,
        "デアル列タ形": 25,
        "デアル列タ系推量形": 26,
        "デアル列タ系省略推量形": 27,
        "デアル列タ系条件形": 28,
        "デアル列タ系連用テ形": 29,
        "デアル列タ系連用タリ形": 30,
        "デス列基本形": 31,
        "デス列音便基本形": 32,
        "デス列基本推量形": 33,
        "デス列音便基本推量形": 34,
        "デス列基本省略推量形": 35,
        "デス列音便基本省略推量形": 36,
        "デス列タ形": 37,
        "デス列タ系推量形": 38,
        "デス列タ系省略推量形": 39,
        "デス列タ系条件形": 40,
        "デス列タ系連用テ形": 41,
        "デス列タ系連用タリ形": 42,
        "ヤ列基本形": 43,
        "ヤ列基本推量形": 44,
        "ヤ列基本省略推量形": 45,
        "ヤ列タ形": 46,
        "ヤ列タ系推量形": 47,
        "ヤ列タ系省略推量形": 48,
        "ヤ列タ系条件形": 49,
        "ヤ列タ系連用タリ形": 50,
    },
    "タル形容詞": {"*": 0, "語幹": 1, "基本形": 2, "基本連用形": 3},
    "判定詞": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "ダ列基本連体形": 3,
        "ダ列特殊連体形": 4,
        "ダ列基本推量形": 5,
        "ダ列基本省略推量形": 6,
        "ダ列基本条件形": 7,
        "ダ列タ形": 8,
        "ダ列タ系推量形": 9,
        "ダ列タ系省略推量形": 10,
        "ダ列タ系条件形": 11,
        "ダ列タ系連用テ形": 12,
        "ダ列タ系連用タリ形": 13,
        "ダ列タ系連用ジャ形": 14,
        "デアル列基本形": 15,
        "デアル列命令形": 16,
        "デアル列基本推量形": 17,
        "デアル列基本省略推量形": 18,
        "デアル列基本条件形": 19,
        "デアル列基本連用形": 20,
        "デアル列タ形": 21,
        "デアル列タ系推量形": 22,
        "デアル列タ系省略推量形": 23,
        "デアル列タ系条件形": 24,
        "デアル列タ系連用テ形": 25,
        "デアル列タ系連用タリ形": 26,
        "デス列基本形": 27,
        "デス列音便基本形": 28,
        "デス列基本推量形": 29,
        "デス列音便基本推量形": 30,
        "デス列基本省略推量形": 31,
        "デス列音便基本省略推量形": 32,
        "デス列タ形": 33,
        "デス列タ系推量形": 34,
        "デス列タ系省略推量形": 35,
        "デス列タ系条件形": 36,
        "デス列タ系連用テ形": 37,
        "デス列タ系連用タリ形": 38,
    },
    "無活用型": {"*": 0, "語幹": 1, "基本形": 2},
    "助動詞ぬ型": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "基本条件形": 3,
        "基本連用形": 4,
        "基本推量形": 5,
        "基本省略推量形": 6,
        "タ形": 7,
        "タ系条件形": 8,
        "タ系連用テ形": 9,
        "タ系推量形": 10,
        "タ系省略推量形": 11,
        "音便基本形": 12,
        "音便推量形": 13,
        "音便省略推量形": 14,
        "文語連体形": 15,
        "文語条件形": 16,
        "文語音便条件形": 17,
    },
    "助動詞だろう型": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "ダ列基本省略推量形": 3,
        "ダ列基本条件形": 4,
        "デアル列基本推量形": 5,
        "デアル列基本省略推量形": 6,
        "デス列基本推量形": 7,
        "デス列音便基本推量形": 8,
        "デス列基本省略推量形": 9,
        "デス列音便基本省略推量形": 10,
        "ヤ列基本推量形": 11,
        "ヤ列基本省略推量形": 12,
    },
    "助動詞そうだ型": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "ダ列タ系連用テ形": 3,
        "デアル列基本形": 4,
        "デス列基本形": 5,
        "デス列音便基本形": 6,
    },
    "助動詞く型": {"*": 0, "語幹": 1, "基本形": 2, "基本連用形": 3, "文語連体形": 4, "文語未然形": 5},
    "動詞性接尾辞ます型": {
        "*": 0,
        "語幹": 1,
        "基本形": 2,
        "未然形": 3,
        "意志形": 4,
        "省略意志形": 5,
        "命令形": 6,
        "タ形": 7,
        "タ系条件形": 8,
        "タ系連用テ形": 9,
        "タ系連用タリ形": 10,
    },
    "動詞性接尾辞うる型": {"*": 0, "語幹": 1, "基本形": 2, "基本条件形": 3},
}
