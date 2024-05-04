import logging
import multiprocessing as mp
import textwrap
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Optional

from cohesion_tools.extractors import PasExtractor
from rhoknp import KNP, Document, Jumanpp, Morpheme, Sentence
from rhoknp.cohesion import RelTag
from rhoknp.props import FeatureDict
from rhoknp.utils.reader import chunk_by_document

from constants import (
    BASE_PHRASE_FEATURES,
    CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID,
    CONJTYPE_TAGS,
    IGNORE_VALUE_FEATURE_PAT,
    POS_TAG2POS_ID,
    POS_TAG_SUBPOS_TAG2SUBPOS_ID,
    SUB_WORD_FEATURES,
)

logging.getLogger("rhoknp").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

UNSUPPORTED_CONJUGATION_FALLBACK_TABLE = {
    ("ナ形容詞", "ダ列文語基本形"): ("ナ形容詞", "ダ列基本連体形"),
    ("判定詞", "ダ列文語連体形"): ("子音動詞ラ行", "基本形"),  # 950112215-023
    ("文語助動詞", "連体形"): ("子音動詞ラ行", "基本形"),
    ("助動詞たり型", "文語連体形"): ("子音動詞ラ行", "基本形"),
    ("助動詞たり文語", "連体形"): ("子音動詞ラ行", "基本形"),
    ("助動詞たり", "文語連体形(たる)"): ("子音動詞ラ行", "基本形"),
    ("助動詞たり", "文語連体形"): ("子音動詞ラ行", "基本形"),
    ("文語", "連体形たる"): ("子音動詞ラ行", "基本形"),
    ("なり列", "古語基本形(なり)"): ("判定詞", "基本形"),
    ("方言", "基本形"): ("判定詞", "基本形"),  # 950114251-001
    ("判定詞", "ヤ列基本形"): ("判定詞", "*"),  # 950115185-003
    ("サ変動詞", "文語已然形"): ("サ変動詞", "*"),  # 950114053-007
    ("ナ形容詞", "語幹異形"): ("ナ形容詞", "語幹"),  # 950115169-035
    ("助動詞そうだ型", "デアル列連用形"): ("助動詞そうだ型", "デアル列基本形"),  # 950115157-027
}

UNSUPPORTED_POS_SUBPOS_FALLBACK_TABLE = {
    ("未定義語", "未対応表現"): ("副詞", "*"),
}


class JumanppAugmenter:
    def __init__(self):
        self.jumanpp = Jumanpp(options=["--partial-input"])

    def augment_document(self, original_document: Document, update_original: bool = True) -> Document:
        buf = ""
        for sentence in original_document.sentences:
            buf += self._create_partial_input(sentence)

        with Popen(self.jumanpp.run_command, stdout=PIPE, stdin=PIPE, encoding="utf-8") as p:
            jumanpp_text, _ = p.communicate(input=buf)
        augmented_document = Document.from_jumanpp(jumanpp_text)

        for original_sentence, augmented_sentence in zip(original_document.sentences, augmented_document.sentences):
            self._postprocess_sentence(original_sentence, augmented_sentence, update_original=update_original)
        return augmented_document

    def augment_sentence(self, original_sentence: Sentence, update_original: bool = True) -> Sentence:
        buf = self._create_partial_input(original_sentence)
        with Popen(self.jumanpp.run_command, stdout=PIPE, stdin=PIPE, encoding="utf-8") as p:
            jumanpp_text, _ = p.communicate(input=buf)
        augmented_sentence = Sentence.from_jumanpp(jumanpp_text)

        self._postprocess_sentence(original_sentence, augmented_sentence, update_original=update_original)
        return augmented_sentence

    @staticmethod
    def _create_partial_input(sentence: Sentence) -> str:
        """
        create raw string for jumanpp --partial-input
        """
        buf = ""
        for morpheme in sentence.morphemes:
            buf += (
                f"\t{morpheme.surf}"
                f"\treading:{morpheme.reading}"
                f"\tbaseform:{morpheme.lemma}"
                f"\tpos:{morpheme.pos}"
                f"\tsubpos:{morpheme.subpos}"
                f"\tconjtype:{morpheme.conjtype}"
                f"\tconjform:{morpheme.conjform}\n"
            )
        buf += "\n"
        return buf

    @staticmethod
    def _postprocess_sentence(
        original_sentence: Sentence,
        augmented_sentence: Sentence,
        update_original: bool = True,
    ) -> None:
        alignment = align_morphemes(original_sentence.morphemes, augmented_sentence.morphemes)
        if alignment is None:
            return None
        keys = []
        for original_morpheme in original_sentence.morphemes:
            keys.append(str(original_morpheme.index))
            if "-".join(keys) in alignment:
                aligned = alignment["-".join(keys)]
                if len(keys) == 1 and len(aligned) == 1:
                    augmented_morpheme = aligned[0]
                    # Jumanpp may override reading
                    augmented_morpheme.reading = original_morpheme.reading
                    if update_original and not original_sentence.is_knp_required():
                        original_morpheme.semantics.update(augmented_morpheme.semantics)
                keys = []


def align_morphemes(morphemes1: list[Morpheme], morphemes2: list[Morpheme]) -> Optional[dict[str, list[Morpheme]]]:
    alignment = {}
    idx1, idx2 = 0, 0
    for _ in range(max(len(morphemes1), len(morphemes2))):
        if idx1 >= len(morphemes1) or idx2 >= len(morphemes2):
            break

        range1 = range(1, min(len(morphemes1) - idx1 + 1, 11))
        range2 = range(1, min(len(morphemes2) - idx2 + 1, 11))
        for i, j in product(range1, range2):
            subseq1, subseq2 = map(
                lambda x: "".join(morpheme.surf for morpheme in x),
                [morphemes1[idx1 : idx1 + i], morphemes2[idx2 : idx2 + j]],
            )
            if subseq1 == subseq2:
                key = "-".join(str(morpheme1.index) for morpheme1 in morphemes1[idx1 : idx1 + i])
                alignment[key] = morphemes2[idx2 : idx2 + j]
                idx1 += i
                idx2 += j
                break
        else:
            return None

    return alignment


def is_target_base_phrase_feature(k: str, v: Any) -> bool:
    name = k + (f":{v}" if isinstance(v, str) and IGNORE_VALUE_FEATURE_PAT.match(k) is None else "")
    return name in BASE_PHRASE_FEATURES


def refresh(document: Document) -> None:
    keys = [feature.split(":")[0] for feature in SUB_WORD_FEATURES]
    for morpheme in document.morphemes:
        feature_dict = FeatureDict()
        if morpheme.base_phrase.head == morpheme:
            feature_dict["基本句-主辞"] = True
        feature_dict.update({key: morpheme.features[key] for key in keys if key in morpheme.features})
        morpheme.features = feature_dict

    keys = [feature.split(":")[0] for feature in BASE_PHRASE_FEATURES]
    for base_phrase in document.base_phrases:
        feature_dict = FeatureDict()
        if (
            (feature := base_phrase.features.get("NE"))
            and isinstance(feature, str)
            and feature.startswith("OPTIONAL") is False
        ):
            feature_dict["NE"] = feature
        feature_dict.update(
            {
                key: base_phrase.features[key]
                for key in keys
                if key in base_phrase.features and is_target_base_phrase_feature(key, base_phrase.features[key])
            },
        )
        base_phrase.features = feature_dict

    for phrase in document.phrases:
        phrase.features.clear()


def assign_features_and_save(
    knp_texts: list[str], output_root: Path, doc_id2split: dict[str, str], restore_coreferring_rels: bool
) -> None:
    jumanpp_augmenter = JumanppAugmenter()
    knp = KNP(options=["-tab", "-dpnd-fast", "-read-feature"])
    for knp_text in knp_texts:
        try:
            document = Document.from_knp(knp_text)
        except ValueError:
            logger.warning("ignore broken knp file")
            continue
        doc_id = document.doc_id
        if doc_id not in doc_id2split:
            continue
        split = doc_id2split[doc_id]

        morpheme_features = []
        unsupported_conjugations: dict[int, tuple[str, int, str, int]] = {}
        unsupported_pos_subpos: dict[int, tuple[str, int, str, int]] = {}
        for morpheme in document.morphemes:
            morpheme_features.append(morpheme.features.copy())
            morpheme.features.clear()
            if fallback_conjugation := UNSUPPORTED_CONJUGATION_FALLBACK_TABLE.get(
                (morpheme.conjtype, morpheme.conjform),
            ):
                unsupported_conjugations[morpheme.global_index] = (
                    morpheme.conjtype,
                    morpheme.conjtype_id,
                    morpheme.conjform,
                    morpheme.conjform_id,
                )
                conjtype, conjform = fallback_conjugation
                morpheme.conjtype = conjtype
                morpheme.conjtype_id = CONJTYPE_TAGS.index(conjtype)
                morpheme.conjform = conjform
                morpheme.conjform_id = CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID[conjtype][conjform]
                logger.info(
                    f"{morpheme.sentence.sid}: replaced unsupported conjugation (type: {morpheme.conjtype}, form: {morpheme.conjform}) with ({conjtype}, {conjform})",
                )
            if fallback_pos_subpos := UNSUPPORTED_POS_SUBPOS_FALLBACK_TABLE.get((morpheme.pos, morpheme.subpos)):
                unsupported_pos_subpos[morpheme.global_index] = (
                    morpheme.pos,
                    morpheme.pos_id,
                    morpheme.subpos,
                    morpheme.subpos_id,
                )
                pos, subpos = fallback_pos_subpos
                morpheme.pos = pos
                morpheme.pos_id = POS_TAG2POS_ID[pos]
                morpheme.subpos = subpos
                morpheme.subpos_id = POS_TAG_SUBPOS_TAG2SUBPOS_ID[pos][subpos]
                logger.info(
                    f"{morpheme.sentence.sid}: replaced unsupported pos/subpos (pos: {morpheme.pos}, subpos: {morpheme.subpos}) with ({pos}, {subpos})",
                )

        # 形態素意味情報付与 (引数に渡したdocumentをupdateする)
        _ = jumanpp_augmenter.augment_document(document)

        # Juman++ によって意味情報フィールド設定されなかった場合は、 KNP で segmentation fault が出ないように NIL を付与
        for morpheme in document.morphemes:
            if not morpheme.semantics:
                morpheme.semantics.nil = True

        # 素性付与
        try:
            document = knp.apply_to_document(document, timeout=300)
        except Exception as e:
            logger.warning(f"{type(e).__name__}: {e}, {document.doc_id}")
            knp = KNP(options=["-tab", "-dpnd-fast", "-read-feature"])
            Path(f"knp_error_{document.doc_id}.knp").write_text(document.to_knp())
            continue

        assert len(document.to_knp().split("\n")) == len(
            knp_text.split("\n"),
        ), f"knp text length mismatch: {document.doc_id}"

        if split == "train":
            # アノテーションの有無を利用して基本句に <非用言格解析:不明> を付与することで解析対象基本句を拡大
            for base_phrase in document.base_phrases:
                if PasExtractor.is_pas_target(base_phrase, verbal=True, nominal=True):
                    continue
                pas_rel_tags = [rel_tag for rel_tag in base_phrase.rel_tags if is_pas_rel_tag(rel_tag)]
                if pas_rel_tags:
                    assert base_phrase.features.get("用言", False) is False
                    base_phrase.features["非用言格解析"] = "不明"

        if restore_coreferring_rels is True:
            # 述語間の共参照関係を利用して割愛されたアノテーションを復元
            # （高畑翔平．「述語項構造解析の改善のためのコーパスアノテーションの分析·補完」．卒業論文，京都大学，2023．）
            document = PasExtractor.restore_pas_annotation(document)

        # 初めから付いていた素性およびKNPサポート外の活用・品詞の付与
        for morpheme, features in zip(document.morphemes, morpheme_features):
            morpheme.features.update(features)
            if conjugation := unsupported_conjugations.get(morpheme.global_index):
                morpheme.conjtype, morpheme.conjtype_id, morpheme.conjform, morpheme.conjform_id = conjugation
                logger.info(
                    f"{morpheme.sentence.sid}: unsupported cojugation (type: {conjugation[0]}, form: {conjugation[2]}) restored",
                )
            if pos_subpos := unsupported_pos_subpos.get(morpheme.global_index):
                morpheme.pos, morpheme.pos_id, morpheme.subpos, morpheme.subpos_id = pos_subpos
                logger.info(
                    f"{morpheme.sentence.sid}: unsupported pos/subpos (pos: {pos_subpos[0]}, subpos: {pos_subpos[2]}) restored",
                )

        refresh(document)

        output_root.joinpath(f"{split}/{doc_id}.knp").write_text(document.to_knp())


def is_pas_rel_tag(rel_tag: RelTag) -> bool:
    # 事態性名詞に付与される傾向にある格
    # カラ格などは「あなたからの手紙」における「手紙」など事態性を持たない名詞に付与されることがあるため除外
    # PAS解析の対象となっている格は加えた方がいいかもしれない
    eventive_rel_types = {"ガ", "ヲ", "ニ", "ガ２", "デ", "ヨリ", "判ガ", "外の関係"}
    eventive_rel_types |= {f"{t}≒" for t in eventive_rel_types}
    return rel_tag.type.startswith("=") is False and rel_tag.type in eventive_rel_types


def test_jumanpp_augmenter():
    jumanpp_augmenter = JumanppAugmenter()

    sentence = Sentence.from_knp(
        textwrap.dedent(
            """\
            # S-ID:w201106-0000060050-1 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-44.94406 MOD:2017/10/15 MEMO:
            * 2D
            + 1D
            コイン こいん コイン 名詞 6 普通名詞 1 * 0 * 0
            + 3D <rel type="ガ" target="不特定:人"/><rel type="ヲ" target="コイン" sid="w201106-0000060050-1" id="0"/>
            トス とす トス 名詞 6 サ変名詞 2 * 0 * 0
            を を を 助詞 9 格助詞 1 * 0 * 0
            * 2D
            + 3D
            ３ さん ３ 名詞 6 数詞 7 * 0 * 0
            回 かい 回 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0
            * -1D
            + -1D <rel type="ガ" target="不特定:人"/><rel type="ガ" mode="？" target="読者"/><rel type="ガ" mode="？" target="著者"/><rel type="ヲ" target="トス" sid="w201106-0000060050-1" id="1"/>
            行う おこなう 行う 動詞 2 * 0 子音動詞ワ行 12 基本形 2
            。 。 。 特殊 1 句点 1 * 0 * 0
            EOS
            """,
        ),
    )
    _ = jumanpp_augmenter.augment_sentence(sentence)
    expected = textwrap.dedent(
        """\
        # S-ID:w201106-0000060050-1 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-44.94406 MOD:2017/10/15 MEMO:
        * 2D
        + 1D
        コイン こいん コイン 名詞 6 普通名詞 1 * 0 * 0 "自動獲得:Wikipedia Wikipediaリダイレクト:硬貨"
        + 3D <rel type="ガ" target="不特定:人"/><rel type="ヲ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        トス とす トス 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:トス/とす ドメイン:スポーツ カテゴリ:抽象物"
        を を を 助詞 9 格助詞 1 * 0 * 0 "代表表記:を/を"
        * 2D
        + 3D
        ３ さん ３ 名詞 6 数詞 7 * 0 * 0 "代表表記:３/さん カテゴリ:数量"
        回 かい 回 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 "代表表記:回/かい 準内容語"
        * -1D
        + -1D <rel type="ガ" target="不特定:人"/><rel type="ガ" mode="？" target="読者"/><rel type="ガ" mode="？" target="著者"/><rel type="ヲ" target="トス" sid="w201106-0000060050-1" id="1"/>
        行う おこなう 行う 動詞 2 * 0 子音動詞ワ行 12 基本形 2 "代表表記:行う/おこなう"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        """,
    )
    assert sentence.to_knp() == expected

    document = Document.from_knp(
        textwrap.dedent(
            """\
            # S-ID:w201106-0000060050-1 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-44.94406 MOD:2017/10/15 MEMO:
            * 2D
            + 1D
            コイン こいん コイン 名詞 6 普通名詞 1 * 0 * 0
            + 3D <rel type="ガ" target="不特定:人"/><rel type="ヲ" target="コイン" sid="w201106-0000060050-1" id="0"/>
            トス とす トス 名詞 6 サ変名詞 2 * 0 * 0
            を を を 助詞 9 格助詞 1 * 0 * 0
            * 2D
            + 3D
            ３ さん ３ 名詞 6 数詞 7 * 0 * 0
            回 かい 回 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0
            * -1D
            + -1D <rel type="ガ" target="不特定:人"/><rel type="ガ" mode="？" target="読者"/><rel type="ガ" mode="？" target="著者"/><rel type="ヲ" target="トス" sid="w201106-0000060050-1" id="1"/>
            行う おこなう 行う 動詞 2 * 0 子音動詞ワ行 12 基本形 2
            。 。 。 特殊 1 句点 1 * 0 * 0
            EOS
            # S-ID:w201106-0000060050-2 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-64.95916 MOD:2013/04/13
            * 1D
            + 1D <rel type="ノ" target="コイン" sid="w201106-0000060050-1" id="0"/>
            表 おもて 表 名詞 6 普通名詞 1 * 0 * 0
            が が が 助詞 9 格助詞 1 * 0 * 0
            * 2D
            + 2D <rel type="ガ" target="表" sid="w201106-0000060050-2" id="0"/><rel type="外の関係" target="数" sid="w201106-0000060050-2" id="2"/>
            出た でた 出る 動詞 2 * 0 母音動詞 1 タ形 10
            * 5D
            + 5D <rel type="ノ" target="出た" sid="w201106-0000060050-2" id="1"/>
            数 かず 数 名詞 6 普通名詞 1 * 0 * 0
            だけ だけ だけ 助詞 9 副助詞 2 * 0 * 0
            、 、 、 特殊 1 読点 2 * 0 * 0
            * 4D
            + 4D
            フィールド ふぃーるど フィールド 名詞 6 普通名詞 1 * 0 * 0
            上 じょう 上 接尾辞 14 名詞性名詞接尾辞 2 * 0 * 0
            の の の 助詞 9 接続助詞 3 * 0 * 0
            * 5D
            + 5D <rel type="修飾" target="フィールド上" sid="w201106-0000060050-2" id="3"/><rel type="修飾" mode="AND" target="数" sid="w201106-0000060050-2" id="2"/>
            モンスター もんすたー モンスター 名詞 6 普通名詞 1 * 0 * 0
            を を を 助詞 9 格助詞 1 * 0 * 0
            * -1D
            + -1D <rel type="ヲ" target="モンスター" sid="w201106-0000060050-2" id="4"/><rel type="ガ" target="不特定:状況"/>
            破壊 はかい 破壊 名詞 6 サ変名詞 2 * 0 * 0
            する する する 動詞 2 * 0 サ変動詞 16 基本形 2
            。 。 。 特殊 1 句点 1 * 0 * 0
            EOS
            """,
        ),
    )
    _ = jumanpp_augmenter.augment_document(document)
    expected = textwrap.dedent(
        """\
        # S-ID:w201106-0000060050-1 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-44.94406 MOD:2017/10/15 MEMO:
        * 2D
        + 1D
        コイン こいん コイン 名詞 6 普通名詞 1 * 0 * 0 "自動獲得:Wikipedia Wikipediaリダイレクト:硬貨"
        + 3D <rel type="ガ" target="不特定:人"/><rel type="ヲ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        トス とす トス 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:トス/とす ドメイン:スポーツ カテゴリ:抽象物"
        を を を 助詞 9 格助詞 1 * 0 * 0 "代表表記:を/を"
        * 2D
        + 3D
        ３ さん ３ 名詞 6 数詞 7 * 0 * 0 "代表表記:３/さん カテゴリ:数量"
        回 かい 回 接尾辞 14 名詞性名詞助数辞 3 * 0 * 0 "代表表記:回/かい 準内容語"
        * -1D
        + -1D <rel type="ガ" target="不特定:人"/><rel type="ガ" mode="？" target="読者"/><rel type="ガ" mode="？" target="著者"/><rel type="ヲ" target="トス" sid="w201106-0000060050-1" id="1"/>
        行う おこなう 行う 動詞 2 * 0 子音動詞ワ行 12 基本形 2 "代表表記:行う/おこなう"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        # S-ID:w201106-0000060050-2 JUMAN:6.1-20101108 KNP:3.1-20101107 DATE:2011/06/21 SCORE:-64.95916 MOD:2013/04/13
        * 1D
        + 1D <rel type="ノ" target="コイン" sid="w201106-0000060050-1" id="0"/>
        表 おもて 表 名詞 6 普通名詞 1 * 0 * 0 "代表表記:表/おもて カテゴリ:場所-機能 漢字読み:訓"
        が が が 助詞 9 格助詞 1 * 0 * 0 "代表表記:が/が"
        * 2D
        + 2D <rel type="ガ" target="表" sid="w201106-0000060050-2" id="0"/><rel type="外の関係" target="数" sid="w201106-0000060050-2" id="2"/>
        出た でた 出る 動詞 2 * 0 母音動詞 1 タ形 10 "代表表記:出る/でる 反義:動詞:入る/はいる 自他動詞:他:出す/だす 補文ト"
        * 5D
        + 5D <rel type="ノ" target="出た" sid="w201106-0000060050-2" id="1"/>
        数 かず 数 名詞 6 普通名詞 1 * 0 * 0 "代表表記:数/かず カテゴリ:数量 漢字読み:訓"
        だけ だけ だけ 助詞 9 副助詞 2 * 0 * 0 "代表表記:だけ/だけ"
        、 、 、 特殊 1 読点 2 * 0 * 0 "代表表記:、/、"
        * 4D
        + 4D
        フィールド ふぃーるど フィールド 名詞 6 普通名詞 1 * 0 * 0 "代表表記:フィールド/ふぃーるど カテゴリ:場所-その他"
        上 じょう 上 接尾辞 14 名詞性名詞接尾辞 2 * 0 * 0 "代表表記:上/じょう"
        の の の 助詞 9 接続助詞 3 * 0 * 0 "代表表記:の/の"
        * 5D
        + 5D <rel type="修飾" target="フィールド上" sid="w201106-0000060050-2" id="3"/><rel type="修飾" mode="AND" target="数" sid="w201106-0000060050-2" id="2"/>
        モンスター もんすたー モンスター 名詞 6 普通名詞 1 * 0 * 0 "代表表記:モンスター/もんすたー カテゴリ:人"
        を を を 助詞 9 格助詞 1 * 0 * 0 "代表表記:を/を"
        * -1D
        + -1D <rel type="ヲ" target="モンスター" sid="w201106-0000060050-2" id="4"/><rel type="ガ" target="不特定:状況"/>
        破壊 はかい 破壊 名詞 6 サ変名詞 2 * 0 * 0 "代表表記:破壊/はかい カテゴリ:抽象物 反義:名詞-サ変名詞:建設/けんせつ"
        する する する 動詞 2 * 0 サ変動詞 16 基本形 2 "代表表記:する/する 自他動詞:自:成る/なる 付属動詞候補（基本）"
        。 。 。 特殊 1 句点 1 * 0 * 0 "代表表記:。/。"
        EOS
        """,
    )
    assert document.to_knp() == expected


def main():
    parser = ArgumentParser()
    parser.add_argument("INPUT", type=str, help="path to input knp dir")
    parser.add_argument("OUTPUT", type=str, help="path to output dir")
    parser.add_argument("--id", type=str, help="path to id")
    parser.add_argument("-j", default=1, type=int, help="number of jobs")
    parser.add_argument(
        "--doc-id-format",
        default="default",
        type=str,
        help="doc id format to identify document boundary",
    )
    args = parser.parse_args()

    knp_texts = []
    for input_file in Path(args.INPUT).glob("**/*.knp"):
        with input_file.open(mode="r") as f:
            knp_texts += [knp_text for knp_text in chunk_by_document(f, doc_id_format=args.doc_id_format)]

    output_root = Path(args.OUTPUT)
    doc_id2split = {}
    for id_file in Path(args.id).glob("*.id"):
        if id_file.stem not in {"train", "dev", "valid", "test"}:
            continue
        split = "valid" if id_file.stem == "dev" else id_file.stem
        output_root.joinpath(split).mkdir(parents=True, exist_ok=True)
        for doc_id in id_file.read_text().splitlines():
            doc_id2split[doc_id] = split

    if args.j > 0:
        chunk_size = len(knp_texts) // args.j + int(len(knp_texts) % args.j > 0)
        iterable = [
            (knp_texts[slice(start, start + chunk_size)], output_root, doc_id2split, True)
            for start in range(0, len(knp_texts), chunk_size)
        ]
        with mp.Pool(args.j) as pool:
            pool.starmap(assign_features_and_save, iterable)
    else:
        assign_features_and_save(knp_texts, output_root, doc_id2split, restore_coreferring_rels=True)


if __name__ == "__main__":
    main()
