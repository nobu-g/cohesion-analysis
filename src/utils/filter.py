import sys
from pathlib import Path
from typing import Union

from rhoknp import Document
from rhoknp.props import FeatureDict

BASE_PHRASE_FEATURES: tuple[str, ...] = (
    # type
    "用言:動",
    "用言:形",
    "用言:判",
    "体言",
    "非用言格解析:動",
    "非用言格解析:形",
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
    # "節-主辞",
    # "節-区切",
)


def filter_doc(document: Document) -> None:
    for phrase in document.phrases:
        phrase.features.clear()
        for base_phrase in phrase.base_phrases:
            features = FeatureDict()
            for key, value in base_phrase.features.items():
                if _item_to_fstring(key, value) in BASE_PHRASE_FEATURES:
                    features[key] = value
            base_phrase.features = features
            for morpheme in base_phrase.morphemes:
                morpheme.features.clear()
                morpheme.semantics.clear()


def _item_to_fstring(key: str, value: Union[str, bool]) -> str:
    if value is False:
        return ""
    if value is True:
        return key
    return f"{key}:{value}"


def main() -> None:
    for path in Path(sys.argv[1]).glob("*.knp"):
        document = Document.from_knp(path.read_text())
        filter_doc(document)
        path.write_text(document.to_knp())


if __name__ == "__main__":
    main()
