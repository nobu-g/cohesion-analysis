def case_analysis_f1_ga(result: dict):
    return result['ガ']['case_analysis'].f1


def case_analysis_f1_wo(result: dict):
    return result['ヲ']['case_analysis'].f1


def case_analysis_f1_ni(result: dict):
    return result['ニ']['case_analysis'].f1


def case_analysis_f1_ga2(result: dict):
    return result['ガ２']['case_analysis'].f1


def case_analysis_f1(result: dict):
    return result['all_case']['case_analysis'].f1


def zero_anaphora_f1_ga(result: dict):
    return result['ガ']['zero_all'].f1


def zero_anaphora_f1_wo(result: dict):
    return result['ヲ']['zero_all'].f1


def zero_anaphora_f1_ni(result: dict):
    return result['ニ']['zero_all'].f1


def zero_anaphora_f1_ga2(result: dict):
    return result['ガ２']['zero_all'].f1


def zero_anaphora_f1(result: dict):
    return result['all_case']['zero_all'].f1


def zero_anaphora_f1_inter(result: dict):
    return result['all_case']['zero_inter_sentential'].f1


def zero_anaphora_f1_intra(result: dict):
    return result['all_case']['zero_intra_sentential'].f1


def zero_anaphora_f1_exophora(result: dict):
    return result['all_case']['zero_exophora'].f1


def coreference_f1(result: dict):
    return result['all_case']['coreference'].f1


def bridging_anaphora_f1(result: dict):
    return result['all_case']['bridging'].f1
