from typing import Optional

from rhoknp import Document
from rhoknp.cohesion import ArgumentType, EndophoraArgument, ExophoraArgument

from datamodule.example.kyoto import Task
from server.schema import OutputData, Phrase, Relation
from utils.util import get_core_expression
from writer.json import DocumentProb, RelProb

BASE_PHRASE_FEATURE_KEYS = {"体言", "用言", "非用言格解析"}

ARG_TYPE_TO_PAS_TYPE = {
    ArgumentType.CASE_EXPLICIT: "overt",
    ArgumentType.CASE_HIDDEN: "dep",
    ArgumentType.OMISSION: "omission",
    ArgumentType.DEMONSTRATIVE: None,
    ArgumentType.EXOPHORA: "exo",
    ArgumentType.UNASSIGNED: None,
}


def _arg_to_pas_type(arg: EndophoraArgument) -> Optional[str]:
    # pas type: overt, dep, intra, inter, exo, None
    if arg.type == ArgumentType.OMISSION:
        if arg.sentence == arg.pas.predicate.base_phrase.sentence:
            return "intra"
        else:
            return "inter"
    else:
        return ARG_TYPE_TO_PAS_TYPE[arg.type]


def gen_response(document: Document, task_to_rel_types: dict[Task, list[str]], prediction: DocumentProb) -> OutputData:
    rel_types: list[str] = sum(task_to_rel_types.values(), [])
    phrases: list[Phrase] = []
    special_tokens = prediction.special_tokens
    special2index = {special: idx for idx, special in enumerate(special_tokens)}
    sid2sidx: dict[str, int] = {sentence.sid: idx for idx, sentence in enumerate(document.sentences)}
    for dtid, base_phrase in enumerate(document.base_phrases):
        rel2prob = {rel_prob.rel: rel_prob for rel_prob in prediction.phrases[dtid].rel_probs}
        relations: list[Relation] = []
        arguments = base_phrase.pas.get_all_arguments(relax=False, include_nonidentical=False)
        relaxed_arguments = base_phrase.pas.get_all_arguments(relax=True, include_nonidentical=True)
        for case, relaxed_args in relaxed_arguments.items():
            args = arguments[case]
            rel_prob: Optional[RelProb] = rel2prob.get(case)
            for arg in relaxed_args:
                if isinstance(arg, EndophoraArgument):
                    relation = Relation(
                        rel_type=case,
                        target=arg.base_phrase.global_index,
                        exo_target="",
                        is_relaxed=(arg not in args),
                        confidence=rel_prob.probs[arg.base_phrase.global_index] if rel_prob is not None else None,
                        pas_type=_arg_to_pas_type(arg),
                        within_sub_doc=True,  # TODO: implement
                    )
                else:
                    # exophora
                    assert isinstance(arg, ExophoraArgument)
                    relation = Relation(
                        rel_type=case,
                        target=None,
                        exo_target=arg.exophora_referent.text,
                        is_relaxed=(arg not in args),
                        confidence=rel_prob.special_probs[special2index[arg.exophora_referent.text]]
                        if rel_prob is not None and arg.exophora_referent.text in special2index
                        else None,
                        pas_type="exo",
                        within_sub_doc=True,
                    )
                relations.append(relation)
        # coreference
        rel_prob = rel2prob.get("=")
        src_mention = base_phrase
        tgt_mentions = src_mention.get_coreferents(include_nonidentical=False)
        relaxed_tgt_mentions = src_mention.get_coreferents(include_nonidentical=True)
        for relaxed_tgt_mention in relaxed_tgt_mentions:
            relations.append(
                Relation(
                    rel_type="=",
                    target=relaxed_tgt_mention.global_index,
                    exo_target="",
                    is_relaxed=(relaxed_tgt_mention not in tgt_mentions),
                    confidence=rel_prob.probs[relaxed_tgt_mention.global_index] if rel_prob is not None else None,
                    pas_type=None,
                    within_sub_doc=True,  # TODO: implement
                ),
            )
        exophora_referents = [
            entity.exophora_referent for entity in src_mention.entities if entity.exophora_referent is not None
        ]
        exophora_referents_relaxed = [
            entity.exophora_referent for entity in src_mention.entities_all if entity.exophora_referent is not None
        ]

        # exophora
        for exophora_referent in exophora_referents_relaxed:
            relations.append(
                Relation(
                    rel_type="=",
                    target=None,
                    exo_target=exophora_referent.text,
                    is_relaxed=(exophora_referent not in exophora_referents),
                    confidence=rel_prob.special_probs[special2index[exophora_referent.text]]
                    if rel_prob is not None and exophora_referent.text in special2index
                    else None,
                    pas_type=None,
                    within_sub_doc=True,
                ),
            )
        phrases.append(
            Phrase(
                sidx=sid2sidx[base_phrase.sentence.sid],
                dtid=base_phrase.global_index,
                surf=base_phrase.text,
                core=get_core_expression(base_phrase),
                targets=sum(
                    [
                        rel_types * bool(base_phrase.features.get(task.value + "対象"))
                        for task, rel_types in task_to_rel_types.items()
                    ],
                    [],
                ),
                relations=relations,
                features=[
                    (k if v is True else f"{k}:{v}")
                    for k, v in base_phrase.features.items()
                    if k in BASE_PHRASE_FEATURE_KEYS
                ],
            ),
        )
    return OutputData(phrases=phrases, rel_types=rel_types, model="deberta-large", knp=document.to_knp())
