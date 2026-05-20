"""
Análise dos dados da CPA/UFT — ciclo 2025 (base primária do TCC).

Este script reproduz a cadeia analítica completa fundamentada nos capítulos
de Fundamentação Teórica e Metodologia do TCC, agora aplicada ao ciclo 2025
da CPA. O ciclo 2024 entra apenas como termo histórico de comparação na
seção temporal (radar 2024 vs 2025 e variações por questão).

Pipeline:

  Parte 1 — Setup, carregamento e limpeza da base 2025
  Parte 2 — Tratamento de respondentes com múltiplos perfis (regra de prioridade)
  Parte 3 — Tipologia das questões e catálogo por eixo SINAES
  Parte 4 — Estatística descritiva (por questão, eixo, segmento e campus)
  Parte 5 — Inferência: Shapiro-Wilk, KW + Dunn + correções, MW, qui², Spearman, Cronbach
  Parte 6 — Comparação temporal 2024 → 2025 sobre as questões mapeadas
  Parte 7 — Análise textual da q70 (frequência, n-gramas, nuvem, categorização por eixo)
  Parte 8 — Síntese e exportação dos artefatos finais

Saídas:
  outputs/2025/*.csv  — tabelas auxiliares para extração direta no LaTeX
  figuras/2025/*.png  — figuras do capítulo de Resultados, com sufixo distinto
                        para não sobrescrever as figuras do ciclo 2024.

Convenções:
  - Escala Likert: 1 (péssimo) a 6 (excelente). "NSO" e vazios → NaN.
  - Segmentos: Discente, Docente, Técnico, Egresso (nessa ordem em todas as figuras).
  - Regra de prioridade para perfis múltiplos: Docente > Técnico > Discente > Egresso.
  - Limiar de significância: α = 0,05. Correções múltiplas: Bonferroni e Benjamini-Hochberg.
"""

from __future__ import annotations

import re
import sys
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
# Agg só quando rodado como script (execução headless do pipeline completo).
# Como módulo importado (notebook), respeita o backend do host (%matplotlib inline etc.).
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    ROOT = Path(__file__).parent
except NameError:
    ROOT = Path.cwd()
BASE_DIR = ROOT / "bases_dados"
OUT_DIR = ROOT / "outputs" / "2025"
FIG_DIR = ROOT / "figuras" / "2025"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PATH_2025 = BASE_DIR / "respostas2025.xlsx"
PATH_2024 = BASE_DIR / "respostas2024.xlsx"
PATH_MAPEAMENTO = BASE_DIR / "mapeamento_2024_2025.csv"

SEGMENTOS_ORDEM = ["Discente", "Docente", "Técnico", "Egresso"]
SEG_PRIORIDADE = ["Docente", "Técnico", "Discente", "Egresso"]

# Okabe & Ito (2008), "Color Universal Design".
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]

SEG_COR = {
    "Discente": "#0072B2",  # azul
    "Docente":  "#D55E00",  # vermelho-laranja
    "Técnico":  "#009E73",  # verde
    "Egresso":  "#CC79A7",  # rosa
}
ANO_COR = {2024: "#56B4E9", 2025: "#0072B2"}

EIXOS_2025 = {
    1: {
        "nome": "Planejamento e Avaliação",
        "binarias": ["q1", "q2", "q3", "q4", "q5", "q6"],
        "condicionais": ["q1.1", "q2.1", "q3.1", "q4.1"],
        "likert": [],
    },
    2: {
        "nome": "Desenvolvimento Institucional",
        "binarias": [],
        "condicionais": [],
        "likert": ["q7", "q8", "q9", "q10", "q11"],
    },
    3: {
        "nome": "Políticas Acadêmicas",
        "binarias": ["q28"],
        "condicionais": ["q28.1"],
        "likert": ["q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19",
                   "q20", "q21", "q22", "q23", "q24", "q25", "q26", "q27",
                   "q29", "q30"],
    },
    4: {
        "nome": "Políticas de Gestão",
        "binarias": [],
        "condicionais": [],
        "likert": ["q31", "q32", "q33", "q34", "q35", "q36", "q37", "q38",
                   "q39", "q40", "q41", "q42"],
    },
    5: {
        "nome": "Infraestrutura Física",
        "binarias": [],
        "condicionais": [],
        "likert": ["q43", "q44", "q45", "q46", "q47", "q48", "q49", "q50",
                   "q51", "q52", "q53", "q54", "q55", "q56", "q57", "q58", "q59"],
    },
}
EXTRAS_RECOMENDACAO = ["q60", "q61"]
TEXTO_LIVRE = "q70"

# Função utilitária pra logar etapas com cabeçalho legível
def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


# ==============================================================================
# Parte 2 — Carregamento, limpeza e tratamento da unidade de análise
# ==============================================================================

def to_likert(v):
    """Converte qualquer valor de célula em uma nota Likert válida (1-6) ou NaN.

    Aceita strings com vírgula decimal, números, "nso", "NA", vazio etc.
    """
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ("", "na", "nso", "n/a", "n", "-"):
        return np.nan
    s = s.replace(",", ".")
    try:
        f = float(s)
    except ValueError:
        return np.nan
    if 1 <= f <= 6:
        return f
    return np.nan


def to_binario(v):
    """Converte resposta sim/não em 1/0 ou NaN."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ("s", "sim", "1", "true", "yes"):
        return 1
    if s in ("n", "nao", "não", "0", "false", "no"):
        return 0
    return np.nan


def carregar_base(path: Path, ano: int) -> pd.DataFrame:
    """Carrega uma planilha de respostas da CPA, normalizando colunas básicas."""
    df = pd.read_excel(path, sheet_name=0)

    # Normaliza nomes de colunas: minúsculo, sem espaços supérfluos
    df.columns = [str(c).strip() for c in df.columns]

    # Algumas bases têm typo "a4.1" no lugar de "q4.1"
    if "a4.1" in df.columns and "q4.1" not in df.columns:
        df = df.rename(columns={"a4.1": "q4.1"})

    # 2025 tem colunas-cabeçalho que entraram no dump (eixo2_como_avalia, eixo3_como_avalia) — descartar
    for col in ("eixo2_como_avalia", "eixo3_como_avalia"):
        if col in df.columns:
            df = df.drop(columns=col)

    # Marca o ano para reuso adiante
    df["__ano__"] = ano

    return df


def _flag_marcado(v) -> bool:
    """True se a célula representa uma marcação válida (não-NaN, não-zero, não-vazio)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    if s in ("", "0", "false", "no", "nao", "não", "n", "nan"):
        return False
    return True


def atribuir_segmento_unico(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a regra de prioridade Docente > Técnico > Discente > Egresso.

    A base bruta tem 4 colunas binárias (PROFESSOR, TECNICO, ALUNO, EGRESSO)
    indicando os perfis marcados pelo respondente. Quando o respondente
    marca mais de um perfil, atribuímos um segmento principal pela regra de
    prioridade — privilegiando o vínculo institucional ativo.
    """
    seg_col = []
    multi_count = 0
    flags_count = {"Docente": 0, "Técnico": 0, "Discente": 0, "Egresso": 0}
    for _, row in df.iterrows():
        flags = {
            "Docente":  _flag_marcado(row.get("PROFESSOR")),
            "Técnico":  _flag_marcado(row.get("TECNICO")),
            "Discente": _flag_marcado(row.get("ALUNO")),
            "Egresso":  _flag_marcado(row.get("EGRESSO")),
        }
        for s, on in flags.items():
            if on:
                flags_count[s] += 1
        n_marcados = sum(flags.values())
        if n_marcados >= 2:
            multi_count += 1
        # Aplica prioridade
        principal = None
        for s in SEG_PRIORIDADE:
            if flags[s]:
                principal = s
                break
        seg_col.append(principal)
    df = df.copy()
    df["segmento"] = seg_col
    print(f"  Marcações brutas (antes da prioridade): {flags_count}")
    print(f"  Respondentes com múltiplos perfis: {multi_count} ({multi_count/len(df)*100:.1f}%)")
    return df


def atribuir_campus(df: pd.DataFrame) -> pd.DataFrame:
    """Determina o campus do respondente a partir das colunas específicas de cada segmento."""
    def pega(row):
        seg = row["segmento"]
        col = {
            "Docente":  "PROFESSOR_CAMPUS",
            "Técnico":  "TECNICO_CAMPUS",
            "Discente": "ALUNO_CAMPUS",
            "Egresso":  "EGRESSO_CAMPUS",
        }.get(seg)
        if col and col in row.index and pd.notna(row[col]):
            return str(row[col]).strip()
        return np.nan
    df = df.copy()
    df["campus"] = df.apply(pega, axis=1)
    return df


def normalizar_likert(df: pd.DataFrame, questoes: list[str]) -> pd.DataFrame:
    """Aplica to_likert nas colunas de questões Likert presentes na base."""
    df = df.copy()
    for q in questoes:
        if q in df.columns:
            df[f"{q}_num"] = df[q].apply(to_likert)
    return df


def normalizar_binarias(df: pd.DataFrame, questoes: list[str]) -> pd.DataFrame:
    """Aplica to_binario nas colunas de questões binárias."""
    df = df.copy()
    for q in questoes:
        if q in df.columns:
            df[f"{q}_bin"] = df[q].apply(to_binario)
    return df


# ==============================================================================
# Parte 3 — Catálogos derivados (Likert universais, scores compostos)
# ==============================================================================

def listar_questoes_likert(eixos=EIXOS_2025) -> list[str]:
    return [q for e in eixos.values() for q in e["likert"]]


def listar_questoes_binarias(eixos=EIXOS_2025) -> list[str]:
    base = [q for e in eixos.values() for q in e["binarias"]]
    return base + EXTRAS_RECOMENDACAO


def listar_questoes_condicionais(eixos=EIXOS_2025) -> list[str]:
    return [q for e in eixos.values() for q in e["condicionais"]]


def determinar_questoes_universais(mapeamento: pd.DataFrame) -> list[str]:
    """Retorna as questões Likert disponíveis para os 4 segmentos (segmentos == '1111')."""
    universais = mapeamento[
        (mapeamento["tipo"] == "Likert") &
        (mapeamento["segmentos"] == "1111")
    ]["q_2025"].tolist()
    return universais


def calcular_score_eixo(df: pd.DataFrame, eixo: int, eixos=EIXOS_2025) -> pd.Series:
    """Score composto = média das questões Likert do eixo, por respondente."""
    cols = [f"{q}_num" for q in eixos[eixo]["likert"] if f"{q}_num" in df.columns]
    if not cols:
        return pd.Series(dtype=float, index=df.index)
    return df[cols].mean(axis=1, skipna=True)


def adicionar_scores_compostos(df: pd.DataFrame, eixos=EIXOS_2025) -> pd.DataFrame:
    df = df.copy()
    for eixo in eixos:
        if eixos[eixo]["likert"]:
            df[f"score_eixo{eixo}"] = calcular_score_eixo(df, eixo, eixos)
    return df


# ==============================================================================
# Parte 4 — Estatística descritiva
# ==============================================================================

def descritiva_por_questao(df: pd.DataFrame, questoes: list[str]) -> pd.DataFrame:
    """Tabela com N, média, mediana, DP e percentual nas faixas 1-2 / 3-4 / 5-6."""
    rows = []
    for q in questoes:
        col = f"{q}_num"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        rows.append({
            "questao": q,
            "eixo": _eixo_da_questao(q),
            "N": int(s.size),
            "media": round(float(s.mean()), 3),
            "mediana": float(s.median()),
            "dp": round(float(s.std(ddof=1)), 3),
            "pct_1_2": round(float((s.le(2)).mean() * 100), 1),
            "pct_3_4": round(float(s.between(3, 4).mean() * 100), 1),
            "pct_5_6": round(float(s.ge(5).mean() * 100), 1),
        })
    return pd.DataFrame(rows).sort_values("media", ascending=False).reset_index(drop=True)


def descritiva_por_eixo(df: pd.DataFrame, eixos=EIXOS_2025) -> pd.DataFrame:
    rows = []
    for eixo, info in eixos.items():
        if not info["likert"]:
            continue
        col = f"score_eixo{eixo}"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        rows.append({
            "eixo": eixo,
            "nome": info["nome"],
            "n_questoes": len(info["likert"]),
            "N": int(s.size),
            "media": round(float(s.mean()), 3),
            "mediana": round(float(s.median()), 3),
            "dp": round(float(s.std(ddof=1)), 3),
        })
    return pd.DataFrame(rows)


def _eixo_da_questao(q: str) -> int | None:
    for eixo, info in EIXOS_2025.items():
        if q in info["likert"] or q in info["binarias"] or q in info["condicionais"]:
            return eixo
    if q in EXTRAS_RECOMENDACAO:
        return None
    return None


# ------------------------------------------------------------------------------
# Cobertura opinativa (NSO): tratamento do "não sei opinar" como dado informativo
# ------------------------------------------------------------------------------

def taxa_nso_por_questao(df_raw: pd.DataFrame, questoes: list[str],
                          segmento_col: str | None = None) -> pd.DataFrame:
    """Para cada questão Likert, calcula:
    - N de respostas válidas (valor numérico 1..6)
    - N de respostas NSO ("não sei opinar")
    - N de respostas em branco (NaN/vazio)
    - % NSO sobre quem respondeu alguma coisa (válido + NSO)

    Opera sobre os valores brutos do DataFrame, antes da normalização Likert,
    porque normalizar_likert colapsa NSO em NaN e perde essa informação.
    Se ``segmento_col`` for informada, quebra a contagem por segmento.
    """
    rows = []
    for q in questoes:
        if q not in df_raw.columns:
            continue
        serie = df_raw[q]
        # Classifica cada célula
        def classificar(v):
            if pd.isna(v):
                return "branco"
            s = str(v).strip().lower()
            if s in ("", "na", "n/a", "n", "-"):
                return "branco"
            if s == "nso":
                return "nso"
            s2 = s.replace(",", ".")
            try:
                f = float(s2)
                if 1 <= f <= 6:
                    return "valido"
            except ValueError:
                pass
            return "branco"

        classes = serie.apply(classificar)

        if segmento_col is not None and segmento_col in df_raw.columns:
            for seg, sub in classes.groupby(df_raw[segmento_col]):
                if pd.isna(seg):
                    continue
                n_val = int((sub == "valido").sum())
                n_nso = int((sub == "nso").sum())
                n_branco = int((sub == "branco").sum())
                total_opinou = n_val + n_nso
                rows.append({
                    "questao": q,
                    "eixo": _eixo_da_questao(q),
                    "segmento": seg,
                    "n_valido": n_val,
                    "n_nso": n_nso,
                    "n_branco": n_branco,
                    "pct_nso": round(100 * n_nso / total_opinou, 1) if total_opinou > 0 else np.nan,
                })
        else:
            n_val = int((classes == "valido").sum())
            n_nso = int((classes == "nso").sum())
            n_branco = int((classes == "branco").sum())
            total_opinou = n_val + n_nso
            rows.append({
                "questao": q,
                "eixo": _eixo_da_questao(q),
                "n_valido": n_val,
                "n_nso": n_nso,
                "n_branco": n_branco,
                "pct_nso": round(100 * n_nso / total_opinou, 1) if total_opinou > 0 else np.nan,
            })
    return pd.DataFrame(rows).sort_values("pct_nso", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------
# Estratificação por campus
# ------------------------------------------------------------------------------

def descritiva_por_campus(df: pd.DataFrame, eixos=EIXOS_2025) -> pd.DataFrame:
    """Médias, medianas e DP dos scores compostos de cada eixo, por campus."""
    rows = []
    for campus, sub in df.groupby("campus"):
        if pd.isna(campus):
            continue
        n_resp = len(sub)
        row = {"campus": campus, "N": n_resp}
        for eixo, info in eixos.items():
            if not info["likert"]:
                continue
            col = f"score_eixo{eixo}"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            row[f"m_eixo{eixo}"] = round(float(s.mean()), 3) if len(s) > 0 else np.nan
            row[f"dp_eixo{eixo}"] = round(float(s.std(ddof=1)), 3) if len(s) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)


def descritiva_por_campus_segmento(df: pd.DataFrame, eixos=EIXOS_2025) -> pd.DataFrame:
    """Média do score composto por eixo, quebrada por campus × segmento.
    Útil para verificar se o padrão Disc < Doc vale em todos os campi."""
    rows = []
    for (campus, seg), sub in df.groupby(["campus", "segmento"]):
        if pd.isna(campus) or pd.isna(seg):
            continue
        n_resp = len(sub)
        if n_resp < 3:  # corte mínimo para não reportar células vazias
            continue
        row = {"campus": campus, "segmento": seg, "N": n_resp}
        for eixo, info in eixos.items():
            if not info["likert"]:
                continue
            col = f"score_eixo{eixo}"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            row[f"m_eixo{eixo}"] = round(float(s.mean()), 3) if len(s) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Estratificação por curso (discentes)
# ------------------------------------------------------------------------------

def descritiva_por_curso_discentes(df: pd.DataFrame, eixos=EIXOS_2025,
                                     min_n: int = 10) -> pd.DataFrame:
    """Médias dos scores compostos por eixo, para cursos de graduação com
    ao menos ``min_n`` discentes respondentes. O valor mínimo filtra cursos
    com amostra insuficiente para leitura descritiva estável."""
    sub_disc = df[df["segmento"] == "Discente"]
    col_curso = "ALUNO_NOME_CURSO_DIPLOMA"
    if col_curso not in sub_disc.columns:
        return pd.DataFrame()
    rows = []
    for curso, sub in sub_disc.groupby(col_curso):
        if pd.isna(curso):
            continue
        n_resp = len(sub)
        if n_resp < min_n:
            continue
        # Limpa prefixo "Curso de " para o nome ficar mais curto
        curso_limpo = str(curso).replace("Curso de ", "").strip()
        row = {"curso": curso_limpo, "N": n_resp}
        for eixo, info in eixos.items():
            if not info["likert"]:
                continue
            col = f"score_eixo{eixo}"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            row[f"m_eixo{eixo}"] = round(float(s.mean()), 3) if len(s) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)



def spearman_inter_eixos(df: pd.DataFrame, eixos=EIXOS_2025) -> pd.DataFrame:
    """Matriz de correlação de Spearman entre os escores compostos dos eixos Likert.

    Mantida apenas para fins de validação interna do agrupamento por eixo:
    a correlação entre escores compostos por eixo confirma que as quatro
    dimensões SINAES estão associadas entre si na percepção dos respondentes.
    A análise de descoberta substantiva, contudo, é feita pela matriz
    questão-a-questão (`spearman_questao_a_questao`), que permite identificar
    pares específicos de itens cujas avaliações se movem juntas e que podem
    indicar relações de dependência ou influência mútua não-óbvias.
    """
    cols = [f"score_eixo{e}" for e, info in eixos.items() if info["likert"] and f"score_eixo{e}" in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    sub = df[cols].dropna()
    rho = sub.corr(method="spearman")
    rho.index = [f"Eixo {c[-1]}" for c in cols]
    rho.columns = [f"Eixo {c[-1]}" for c in cols]
    return rho.round(4)


def spearman_questao_a_questao(df: pd.DataFrame, questoes: list[str]) -> pd.DataFrame:
    """Matriz de correlação de Spearman para todos os pares de questões Likert.

    Calcula o coeficiente $\\rho$ de Spearman entre cada par possível de
    questões da lista informada (geralmente as questões universais), usando
    apenas os respondentes que têm valores válidos nas duas questões do par
    (correlação par a par com `pairwise.complete.obs`). O resultado é uma
    matriz simétrica $k \\times k$, onde $k$ é o número de questões válidas.

    Esta é a análise de descoberta substantiva: dela saem os pares de
    questões cujas avaliações se movem juntas com força (potenciais
    relações de dependência ou influência) e os pares que se movem em
    independência aparente.
    """
    cols_validas = [f"{q}_num" for q in questoes if f"{q}_num" in df.columns]
    if len(cols_validas) < 2:
        return pd.DataFrame()
    sub = df[cols_validas]
    rho = sub.corr(method="spearman", min_periods=30)
    # Renomeia para os códigos curtos das questões (sem o sufixo _num)
    nomes = [c.replace("_num", "") for c in cols_validas]
    rho.index = nomes
    rho.columns = nomes
    return rho.round(4)


def spearman_questao_a_questao_estratificado(
    df: pd.DataFrame,
    questoes: list[str],
    segmentos: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Calcula a matriz de Spearman questão-a-questão SEPARADAMENTE para cada
    segmento institucional informado. Retorna um dicionário {segmento: matriz}.

    Esta é a verificação contra o paradoxo de Simpson (Simpson, 1951): se o
    cluster de correlações observado no agregado de todos os respondentes
    sobrevive ao cálculo dentro de cada subgrupo individualmente, então ele
    é uma estrutura real da percepção institucional, não um artefato da
    agregação de segmentos heterogêneos. Se o cluster desaparece dentro dos
    subgrupos, a correlação no agregado era espúria e deve ser descartada
    como achado de descoberta. A discussão conceitual está em
    Fundamentação Teórica, seção `ssec:simpson`.
    """
    if segmentos is None:
        segmentos = ["Discente", "Docente", "Técnico"]
    out = {}
    for seg in segmentos:
        sub = df[df["segmento"] == seg]
        if sub.empty:
            continue
        out[seg] = spearman_questao_a_questao(sub, questoes)
    return out


def tabela_estratificacao_por_segmento(
    rho_agregado: pd.DataFrame,
    rho_por_segmento: dict[str, pd.DataFrame],
    n: int = 20,
) -> pd.DataFrame:
    """Para os top N pares do agregado por |ρ|, recalcula o coeficiente em
    cada segmento e gera tabela com média e desvio padrão entre segmentos.

    A comparação entre o ρ agregado e os ρ dentro de cada segmento permite
    identificar pares cuja correlação no agregado se sustenta quando os
    segmentos são analisados separadamente (padrão estável) e pares cuja
    correlação agregada desaparece ou enfraquece intra-segmento (padrão
    que pode estar sendo puxado pela heterogeneidade entre segmentos).

    Colunas retornadas:
    - q1, q2, rho_agregado
    - rho_<segmento> para cada segmento informado
    - rho_media_segmentos: média dos rhos intra-segmento
    - rho_desvio_segmentos: desvio padrão dos rhos intra-segmento
    - delta_agregado_segmentos: rho_agregado - rho_media_segmentos
      (positivo = correlação maior no agregado do que na média intra-segmento;
      próximo de zero = correlação estável entre os níveis de análise)
    """
    top = top_pares_spearman(rho_agregado, n=n)
    if top.empty:
        return pd.DataFrame()
    rows = []
    for _, r in top.iterrows():
        q1, q2 = r["q1"], r["q2"]
        registro = {
            "q1": q1,
            "q2": q2,
            "rho_agregado": float(r["rho"]),
        }
        valores = []
        for seg, mat in rho_por_segmento.items():
            if q1 in mat.index and q2 in mat.columns:
                v = mat.loc[q1, q2]
                if pd.notna(v):
                    registro[f"rho_{seg}"] = round(float(v), 4)
                    valores.append(float(v))
                else:
                    registro[f"rho_{seg}"] = np.nan
            else:
                registro[f"rho_{seg}"] = np.nan
        if valores:
            registro["rho_media_segmentos"] = round(float(np.mean(valores)), 4)
            registro["rho_desvio_segmentos"] = round(float(np.std(valores, ddof=0)), 4)
            registro["delta_agregado_segmentos"] = round(
                float(r["rho"]) - float(np.mean(valores)), 4
            )
        else:
            registro["rho_media_segmentos"] = np.nan
            registro["rho_desvio_segmentos"] = np.nan
            registro["delta_agregado_segmentos"] = np.nan
        rows.append(registro)
    return pd.DataFrame(rows)


def top_pares_spearman(matriz: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Extrai os N pares de questões com maior $|\\rho|$ de uma matriz simétrica.

    Remove a diagonal (correlação de uma questão consigo mesma) e duplicatas
    (matriz é simétrica), e ordena pelo valor absoluto do coeficiente. Útil
    para identificar pares substantivamente correlacionados que merecem
    inspeção qualitativa.
    """
    if matriz.empty:
        return pd.DataFrame()
    pares = []
    cols = list(matriz.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = matriz.iloc[i, j]
            if pd.isna(rho):
                continue
            pares.append({
                "q1": cols[i],
                "q2": cols[j],
                "rho": float(rho),
                "abs_rho": abs(float(rho)),
            })
    if not pares:
        return pd.DataFrame()
    out = pd.DataFrame(pares).sort_values("abs_rho", ascending=False).head(n)
    return out.drop(columns=["abs_rho"]).reset_index(drop=True)


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Calcula o Alpha de Cronbach a partir de uma matriz de itens (linhas=resp, cols=questões)."""
    items = items.dropna()
    k = items.shape[1]
    if k < 2 or items.empty:
        return np.nan
    var_items = items.var(axis=0, ddof=1).sum()
    var_total = items.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return np.nan
    return float(k / (k - 1) * (1 - var_items / var_total))


def cronbach_por_eixo(df: pd.DataFrame, eixos=EIXOS_2025) -> pd.DataFrame:
    rows = []
    for eixo, info in eixos.items():
        cols = [f"{q}_num" for q in info["likert"] if f"{q}_num" in df.columns]
        if len(cols) < 2:
            continue
        items = df[cols]
        n_completos = int(items.dropna().shape[0])
        alpha = cronbach_alpha(items)
        rows.append({
            "eixo": eixo,
            "nome": info["nome"],
            "n_itens": len(cols),
            "N_completos": n_completos,
            "alpha": round(float(alpha), 4) if not pd.isna(alpha) else np.nan,
            "interpretacao": _classificar_alpha(alpha),
        })
    return pd.DataFrame(rows)


def _classificar_alpha(a: float) -> str:
    if pd.isna(a):
        return "—"
    if a < 0.50:
        return "Inaceitável"
    if a < 0.60:
        return "Pobre"
    if a < 0.70:
        return "Questionável"
    if a < 0.80:
        return "Aceitável"
    if a < 0.90:
        return "Bom"
    return "Excelente"


# ==============================================================================
# Parte 6 — Comparação temporal 2024 → 2025
# ==============================================================================

def carregar_mapeamento() -> pd.DataFrame:
    # `segmentos` é uma sequência de 4 dígitos como "1111" / "0100" — força string
    # com padding pra preservar zeros à esquerda, que pandas perderia se lesse como int.
    mapa = pd.read_csv(PATH_MAPEAMENTO, encoding="utf-8", dtype={"segmentos": str})
    mapa["segmentos"] = mapa["segmentos"].fillna("").str.zfill(4)
    return mapa


def comparacao_temporal(df_2024: pd.DataFrame, df_2025: pd.DataFrame, mapa: pd.DataFrame) -> pd.DataFrame:
    """Para cada questão Likert comparável entre 2024 e 2025, calcula
    a média em cada ciclo e o delta descritivo (2025 - 2024)."""
    likert_comp = mapa[
        (mapa["tipo"] == "Likert") &
        (mapa["comparavel"].isin(["Sim", "Parcial"])) &
        (mapa["q_2024"].notna()) &
        (mapa["q_2024"] != "")
    ]
    rows = []
    for _, m in likert_comp.iterrows():
        q25 = m["q_2025"]
        q24 = m["q_2024"]
        col_25 = f"{q25}_num"
        col_24 = f"{q24}_num"
        if col_25 not in df_2025.columns or col_24 not in df_2024.columns:
            continue
        a = df_2024[col_24].dropna().values
        b = df_2025[col_25].dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        rows.append({
            "q_2025": q25,
            "q_2024": q24,
            "eixo": int(m["eixo"]),
            "comparavel": m["comparavel"],
            "n_2024": len(a),
            "n_2025": len(b),
            "media_2024": round(float(np.mean(a)), 3),
            "media_2025": round(float(np.mean(b)), 3),
            "delta": round(float(np.mean(b) - np.mean(a)), 3),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("delta", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def medias_eixo_por_ano(df: pd.DataFrame, eixos=EIXOS_2025) -> dict[int, float]:
    out = {}
    for eixo, info in eixos.items():
        if not info["likert"]:
            continue
        col = f"score_eixo{eixo}"
        if col in df.columns:
            out[eixo] = float(df[col].dropna().mean())
    return out


# ==============================================================================
# Parte 7 — Análise textual da q70 (campo aberto)
# ==============================================================================

def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


_STOPWORDS_RAW = {
    # genéricas
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "no", "na", "nos", "nas", "em", "para", "por", "com", "sem", "sob", "sobre",
    "que", "se", "ao", "à", "às", "aos", "pelo", "pela", "pelos", "pelas",
    "e", "ou", "mas", "como", "quando", "onde", "porque", "porquê", "pois",
    "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "isto", "isso", "aquilo", "aquele", "aquela", "aqueles", "aquelas",
    "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas", "me", "te",
    "nos", "vos", "lhe", "lhes", "meu", "minha", "seu", "sua", "nosso", "nossa",
    "ser", "estar", "ter", "haver", "fazer", "ir", "vir", "ver", "dar", "dizer",
    "é", "são", "foi", "foram", "será", "serão", "tem", "têm", "tinha", "tinham",
    "está", "estão", "esteve", "estiveram", "há", "havia", "houve",
    "muito", "muitos", "muita", "muitas", "pouco", "poucos", "pouca", "poucas",
    "mais", "menos", "também", "ainda", "já", "sempre", "nunca", "todo", "toda",
    "todos", "todas", "outro", "outra", "outros", "outras", "mesmo", "mesma",
    "tudo", "nada", "algo", "alguém", "ninguém", "qual", "quais", "qualquer",
    "não", "sim", "talvez", "porém", "contudo", "entretanto", "assim", "então",
    "só", "apenas", "bem", "mal", "etc", "ex", "alem", "além",
    "ja", "pode", "podem", "deve", "devem", "deveria", "deveriam",
    "alguns", "algumas", "algum", "alguma", "varios", "várias", "varias", "vários",
    "fica", "ficam", "ficar", "ficou", "fica", "vai", "vem", "tao", "tão",
    # contextuais (genéricas demais para o tema)
    "uft", "campus", "universidade",
}
# Como o normalizador remove acentos antes de tokenizar, todas as stopwords precisam
# existir também na versão sem acento — caso contrário "nao" / "sao" passam batido.
STOPWORDS_PT = _STOPWORDS_RAW | {_strip_accents(w) for w in _STOPWORDS_RAW}


def normalizar_texto(s: str) -> str:
    s = s.lower()
    # remove acentos para reduzir variação
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # mantém só letras e espaços
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenizar(s: str) -> list[str]:
    return [t for t in normalizar_texto(s).split() if len(t) > 2 and t not in STOPWORDS_PT]


def analise_q70(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame]:
    """Retorna (estatisticas, top_termos, bigramas, categorizacao_por_eixo)."""
    if TEXTO_LIVRE not in df.columns:
        return {}, pd.DataFrame(), {}, pd.DataFrame()
    sub = df[[TEXTO_LIVRE, "segmento"]].copy()
    sub = sub[sub[TEXTO_LIVRE].notna() & (sub[TEXTO_LIVRE].astype(str).str.strip() != "")]
    sub["len_palavras"] = sub[TEXTO_LIVRE].astype(str).str.split().str.len()

    estat = {
        "total_respondentes": int(df.shape[0]),
        "com_comentario": int(sub.shape[0]),
        "pct_com_comentario": round(sub.shape[0] / df.shape[0] * 100, 1),
        "media_palavras": round(float(sub["len_palavras"].mean()), 1) if not sub.empty else 0,
    }
    estat["por_segmento"] = (
        df.groupby("segmento")[TEXTO_LIVRE]
        .apply(lambda s: int(s.notna().sum()))
        .to_dict()
    )

    # Frequência de termos
    todos_tokens: list[str] = []
    for txt in sub[TEXTO_LIVRE].astype(str):
        todos_tokens.extend(tokenizar(txt))
    contagem = Counter(todos_tokens)
    top = pd.DataFrame(contagem.most_common(30), columns=["termo", "freq"])

    # Bigramas
    bigramas: Counter = Counter()
    for txt in sub[TEXTO_LIVRE].astype(str):
        toks = tokenizar(txt)
        for i in range(len(toks) - 1):
            bigramas[(toks[i], toks[i + 1])] += 1
    top_bigramas = pd.DataFrame(
        [(" ".join(k), v) for k, v in bigramas.most_common(20)],
        columns=["bigrama", "freq"],
    )

    # Categorização por dicionário temático alinhado aos eixos
    DIC = {
        2: ["formacao", "mercado", "trabalho", "ensino", "estudante", "estudantes",
            "auxilio", "auxilios", "permanencia", "afirmativa", "afirmativas", "social"],
        3: ["aula", "aulas", "curso", "cursos", "ensino", "pesquisa", "extensao",
            "professor", "professores", "ava", "moodle", "plataforma", "educacao",
            "ouvidoria", "comunicacao", "site", "portal", "redes", "psicopedagogico"],
        4: ["gestao", "direcao", "coordenacao", "transparencia", "recursos",
            "egresso", "egressos", "internacionalizacao", "indicadores", "instancias",
            "pro reitoria", "pro-reitoria"],
        5: ["sala", "salas", "laboratorio", "laboratorios", "internet", "biblioteca",
            "auditorio", "sanitaria", "sanitarias", "banheiro", "cantina", "restaurante",
            "convivencia", "limpeza", "seguranca", "acessibilidade", "transporte",
            "onibus", "infraestrutura", "estacionamento", "ar condicionado",
            "bloco", "espaco", "espacos"],
    }
    cat_rows = []
    for eixo, palavras in DIC.items():
        n_coment = 0
        for txt in sub[TEXTO_LIVRE].astype(str):
            t = normalizar_texto(txt)
            if any(p in t for p in palavras):
                n_coment += 1
        cat_rows.append({
            "eixo": eixo,
            "nome": EIXOS_2025[eixo]["nome"],
            "n_comentarios": n_coment,
            "pct_do_total": round(n_coment / max(sub.shape[0], 1) * 100, 1),
        })
    categorizacao = pd.DataFrame(cat_rows).sort_values("n_comentarios", ascending=False).reset_index(drop=True)

    return estat, top, top_bigramas, categorizacao


# ==============================================================================
# Parte 8 — Geração de figuras
# ==============================================================================

# ------------------------------------------------------------------------------
# Estilo de publicação científica.
#
# Diretrizes adotadas (alinhadas com o que já é praxe no notebook original do
# trabalho, em Okabe & Ito 2008 e nas diretrizes da Nature/Science para figuras):
#
#   - Resolução de exportação: 300 DPI (PNG + PDF vetorial)
#   - Fonte: Arial / Helvetica / DejaVu Sans (sem serifa, ≥ 9 pt)
#   - Spines superior e direita removidas; eixos finos (0.8 pt)
#   - Sem grid de fundo por padrão; legendas sem moldura
#   - Paleta Okabe-Ito (acessível para daltônicos) como ciclo de cores padrão
#   - Fundo branco em todas as figuras (compatível com impressão e transparências)
#
# Toda figura é salva pela função `salvar_fig()` em PNG (raster, alta-DPI)
# e PDF (vetorial, escalável sem perda) — o LaTeX deve referenciar a versão
# que melhor servir ao contexto.
# ------------------------------------------------------------------------------

PUB_PARAMS = {
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "figure.figsize": (7, 4.5),
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "xtick.major.size": 4,
    "xtick.major.width": 0.8,
    "xtick.labelsize": 9,
    "ytick.major.size": 4,
    "ytick.major.width": 0.8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.facecolor": "white",
    "image.cmap": "viridis",
}
plt.rcParams.update(PUB_PARAMS)
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=OKABE_ITO)
sns.set_theme(style="ticks", context="paper", font_scale=1.15,
              rc={"axes.spines.top": False, "axes.spines.right": False})


KIT_DIR = FIG_DIR / "kit_cpa"
KIT_DIR.mkdir(parents=True, exist_ok=True)


def salvar_fig(fig, nome: str, formatos=("png",), subdir: Path | None = None) -> Path:
    """Salva uma figura em 300 DPI, fundo branco, apenas PNG.

    Se subdir for fornecido, salva lá em vez de FIG_DIR.
    Retorna o Path do arquivo salvo.
    """
    destino = subdir if subdir is not None else FIG_DIR
    destino.mkdir(parents=True, exist_ok=True)
    out = None
    for fmt in formatos:
        caminho = destino / f"{nome}.{fmt}"
        fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
        if fmt == "png":
            out = caminho
    return out


def fig_media_por_eixo(desc_eixos: pd.DataFrame, ano: int = 2025) -> Path:
    """Médias dos escores compostos por eixo."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    eixos = [f"Eixo {e}" for e in desc_eixos["eixo"]]
    medias = desc_eixos["media"].values
    cor = OKABE_ITO[4]  # azul
    ax.plot(eixos, medias, "o", color=cor, markersize=8,
            markeredgecolor="white", markeredgewidth=0.8)
    for x, y in zip(eixos, medias):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, color="black")
    ax.set_ylim(3.0, 5.5)
    ax.set_ylabel("Escore composto (escala 1--6)")
    ax.set_xlabel("")
    sns.despine(ax=ax)
    out = salvar_fig(fig, f"media_eixos_{ano}")
    plt.close(fig)
    return out


def fig_boxplot_por_eixo(df: pd.DataFrame, ano: int = 2025) -> Path:
    """Distribuição do percentual de satisfação por eixo, com pontos individuais.

    Em vez de plotar a média Likert bruta (1–6), apresenta a métrica em
    porcentagem de satisfação por respondente: pct = (média_eixo - 1) / 5 * 100,
    onde 0% corresponde à pior nota possível (1) e 100% à melhor (6). A
    transformação é linear, preserva a ordem e os intervalos relativos, e dá ao
    leitor uma escala intuitiva.

    Sobre cada caixa do boxplot é desenhado um strip plot (pontos individuais
    com jitter horizontal) representando o percentual de satisfação de cada
    respondente. Os pontos são desenhados com zorder maior do que o boxplot
    para ficarem à frente da caixa, de modo que o leitor perceba ao mesmo
    tempo (a) a distribuição resumida pelo boxplot e (b) a dispersão real das
    observações individuais.
    """
    eixos_validos = [e for e in EIXOS_2025 if EIXOS_2025[e]["likert"] and f"score_eixo{e}" in df.columns]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    cor_box = OKABE_ITO[1]      # azul claro (preenchimento)
    cor_borda = OKABE_ITO[4]    # azul forte (borda)
    cor_mediana = OKABE_ITO[5]  # vermelho-laranja
    cor_pontos = OKABE_ITO[4]   # azul forte para pontos

    dados_pct = []
    for e in eixos_validos:
        score = df[f"score_eixo{e}"].dropna().values
        pct = (score - 1) / 5.0 * 100.0  # transforma 1–6 em 0–100%
        dados_pct.append(pct)

    posicoes = np.arange(1, len(eixos_validos) + 1)

    # Boxplot por trás (zorder baixo)
    bp = ax.boxplot(
        dados_pct, positions=posicoes, widths=0.55, patch_artist=True,
        showfliers=False,  # outliers vão aparecer pelos pontos individuais
        zorder=1,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(cor_box); patch.set_edgecolor(cor_borda); patch.set_linewidth(1.1)
        patch.set_alpha(0.65)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(cor_borda); w.set_linewidth(1.0)
    for m in bp["medians"]:
        m.set_color(cor_mediana); m.set_linewidth(1.8)

    # Strip plot na frente (zorder alto). Jitter horizontal pequeno.
    for pos, pct in zip(posicoes, dados_pct):
        jitter = rng.uniform(-0.18, 0.18, size=len(pct))
        ax.scatter(pos + jitter, pct, s=8, color=cor_pontos, alpha=0.30,
                   edgecolors="none", zorder=3)

    # Rótulo do eixo X traz o número de itens de cada eixo. Isso é importante
    # porque a granularidade do percentual de satisfação por respondente é
    # 20/k pontos percentuais (onde k é o número de itens do eixo): com k
    # pequeno, a métrica só pode assumir poucos valores discretos, e os
    # pontos do strip plot se alinham em "listras" horizontais. Tornar o k
    # visível no eixo dá ao leitor a chave para interpretar essa aparência.
    ax.set_xticks(posicoes)
    ax.set_xticklabels(
        [f"Eixo {e}\n({len(EIXOS_2025[e]['likert'])} itens)" for e in eixos_validos],
        fontsize=10,
    )
    ax.set_ylabel(r"Satisfação por respondente (\%)")
    ax.set_xlabel("")
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="0.92", linewidth=0.6, zorder=0)
    sns.despine(ax=ax)
    out = salvar_fig(fig, f"boxplot_eixos_{ano}")
    plt.close(fig)
    return out


def fig_radar_segmentos(df: pd.DataFrame, ano: int = 2025) -> Path:
    """Radar comparando perfil dos quatro segmentos nos eixos Likert."""
    eixos_validos = [e for e in EIXOS_2025 if EIXOS_2025[e]["likert"]]
    labels = [f"Eixo {e}" for e in eixos_validos]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for seg in SEGMENTOS_ORDEM:
        sub = df[df["segmento"] == seg]
        valores = []
        for e in eixos_validos:
            col = f"score_eixo{e}"
            v = float(sub[col].dropna().mean()) if col in sub else np.nan
            valores.append(v)
        valores += valores[:1]
        ax.plot(angles, valores, "o-", color=SEG_COR[seg], label=seg, lw=1.8,
                markersize=5, markeredgecolor="white", markeredgewidth=0.7)
        ax.fill(angles, valores, color=SEG_COR[seg], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(2, 6)
    ax.set_yticks([3, 4, 5, 6])
    ax.set_yticklabels(["3", "4", "5", "6"], fontsize=8)
    ax.tick_params(axis="y", pad=2)
    ax.spines["polar"].set_linewidth(0.5)
    ax.grid(color="0.85", linewidth=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.08), frameon=False)
    out = salvar_fig(fig, f"radar_segmentos_{ano}")
    plt.close(fig)
    return out


def fig_radar_campus(df: pd.DataFrame, ano: int = 2025,
                      min_n: int = 30) -> Path:
    """Radar comparando perfil por campus nos eixos Likert.

    Apenas campi com ao menos ``min_n`` respondentes entram na figura, para
    evitar polígonos instáveis baseados em amostras minúsculas (ex.: Porto
    Nacional com 15 respondentes ou Reitoria com 11).
    """
    eixos_validos = [e for e in EIXOS_2025 if EIXOS_2025[e]["likert"]]
    labels = [f"Eixo {e}" for e in eixos_validos]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Paleta Okabe-Ito para campi (cores distintas e colorblind-safe)
    campi_ordem = (df["campus"]
                     .value_counts()
                     .loc[lambda s: s >= min_n]
                     .index.tolist())
    paleta_campi = dict(zip(campi_ordem, OKABE_ITO[:len(campi_ordem)]))

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for campus in campi_ordem:
        sub = df[df["campus"] == campus]
        valores = []
        for e in eixos_validos:
            col = f"score_eixo{e}"
            v = float(sub[col].dropna().mean()) if col in sub else np.nan
            valores.append(v)
        valores += valores[:1]
        cor = paleta_campi[campus]
        ax.plot(angles, valores, "o-", color=cor, label=f"{campus} (n={len(sub)})",
                lw=1.8, markersize=5, markeredgecolor="white", markeredgewidth=0.7)
        ax.fill(angles, valores, color=cor, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(2, 6)
    ax.set_yticks([3, 4, 5, 6])
    ax.set_yticklabels(["3", "4", "5", "6"], fontsize=8)
    ax.tick_params(axis="y", pad=2)
    ax.spines["polar"].set_linewidth(0.5)
    ax.grid(color="0.85", linewidth=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), frameon=False,
              fontsize=9)
    out = salvar_fig(fig, f"radar_campus_{ano}")
    plt.close(fig)
    return out


def fig_heatmap_questao_campus(df: pd.DataFrame, questoes: list[str],
                                 ano: int = 2025, min_n: int = 30) -> Path:
    """Heatmap das médias por questão × campus, para as questões informadas."""
    campi = (df["campus"]
               .value_counts()
               .loc[lambda s: s >= min_n]
               .index.tolist())
    if not campi or not questoes:
        return FIG_DIR / f"heatmap_questao_campus_{ano}.png"

    mat = []
    linhas = []
    for q in questoes:
        col = f"{q}_num"
        if col not in df.columns:
            continue
        linha = []
        for c in campi:
            s = df.loc[df["campus"] == c, col].dropna()
            linha.append(float(s.mean()) if len(s) > 0 else np.nan)
        mat.append(linha)
        linhas.append(q)
    if not mat:
        return FIG_DIR / f"heatmap_questao_campus_{ano}.png"

    arr = np.array(mat)
    fig, ax = plt.subplots(figsize=(1.5 + 0.9 * len(campi), 0.28 * len(linhas) + 1))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=2.5, vmax=5.5)
    ax.set_xticks(range(len(campi)))
    ax.set_xticklabels([f"{c}\n(n={(df['campus']==c).sum()})" for c in campi],
                        fontsize=9)
    ax.set_yticks(range(len(linhas)))
    ax.set_yticklabels(linhas, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Média (escala 1--6)", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    out = salvar_fig(fig, f"heatmap_questao_campus_{ano}")
    plt.close(fig)
    return out


def fig_radar_2024_2025(med_2024: dict, med_2025: dict) -> Path:
    """Radar comparativo 2024 vs 2025 — peça-chave da seção temporal."""
    eixos_comuns = sorted(set(med_2024) & set(med_2025))
    labels = [f"Eixo {e}" for e in eixos_comuns]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    v24 = [med_2024[e] for e in eixos_comuns] + [med_2024[eixos_comuns[0]]]
    v25 = [med_2025[e] for e in eixos_comuns] + [med_2025[eixos_comuns[0]]]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.plot(angles, v24, "o-", color=ANO_COR[2024], label="2024 (estado anterior)",
            lw=2.0, markersize=6, markeredgecolor="white", markeredgewidth=0.8)
    ax.fill(angles, v24, color=ANO_COR[2024], alpha=0.15)
    ax.plot(angles, v25, "o-", color=ANO_COR[2025], label="2025 (estado atual)",
            lw=2.0, markersize=6, markeredgecolor="white", markeredgewidth=0.8)
    ax.fill(angles, v25, color=ANO_COR[2025], alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(2, 6)
    ax.set_yticks([3, 4, 5, 6])
    ax.set_yticklabels(["3", "4", "5", "6"], fontsize=8)
    ax.tick_params(axis="y", pad=2)
    ax.spines["polar"].set_linewidth(0.5)
    ax.grid(color="0.85", linewidth=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08), frameon=False)
    out = salvar_fig(fig, "radar_2024_vs_2025")
    plt.close(fig)
    return out


def fig_gap_eixos_2024_2025(med_2024: dict, med_2025: dict) -> Path:
    """Variação 2024 → 2025 da média por eixo, em colunas verticais agrupadas.

    Para cada eixo aparecem duas colunas lado a lado: a média do ciclo 2024
    (em tom claro) e a média do ciclo 2025 (em tom escuro). A variação é lida
    diretamente pela diferença de altura entre as duas colunas, e o
    deslocamento absoluto fica anotado acima das colunas. Esse formato dá ao
    leitor a percepção de variação muito mais imediata do que barras
    horizontais com o delta isolado.
    """
    eixos_comuns = sorted(set(med_2024) & set(med_2025))
    n = len(eixos_comuns)
    posicoes = np.arange(n)
    largura = 0.36

    v24 = np.array([med_2024[e] for e in eixos_comuns])
    v25 = np.array([med_2025[e] for e in eixos_comuns])
    deltas = v25 - v24

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    cor_2024 = ANO_COR[2024]
    cor_2025 = ANO_COR[2025]
    cor_delta_pos = OKABE_ITO[2]  # verde
    cor_delta_neg = OKABE_ITO[5]  # vermelho-laranja

    bars_24 = ax.bar(posicoes - largura / 2, v24, width=largura,
                     color=cor_2024, edgecolor="white", linewidth=0.6,
                     label="2024 (estado anterior)", zorder=2)
    bars_25 = ax.bar(posicoes + largura / 2, v25, width=largura,
                     color=cor_2025, edgecolor="white", linewidth=0.6,
                     label="2025 (estado atual)", zorder=2)

    # Anota o valor de cada coluna na ponta
    for b in bars_24:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                fontsize=8, color="0.30")
    for b in bars_25:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                fontsize=8, color="0.10", fontweight="bold")

    # Anota o delta acima do par de colunas
    altura_max = max(v24.max(), v25.max())
    for i, d in enumerate(deltas):
        cor = cor_delta_pos if d >= 0 else cor_delta_neg
        sinal = "+" if d >= 0 else ""
        ax.text(posicoes[i], altura_max + 0.45, f"$\\Delta = {sinal}{d:.2f}$",
                ha="center", va="bottom", fontsize=9, color=cor, fontweight="bold")

    ax.set_xticks(posicoes)
    ax.set_xticklabels([f"Eixo {e}\n{EIXOS_2025[e]['nome'].split()[0]}" for e in eixos_comuns],
                       fontsize=9)
    ax.set_ylabel("Média do escore composto (escala 1--6)")
    ax.set_ylim(0, altura_max + 1.1)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.grid(axis="y", color="0.92", linewidth=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    sns.despine(ax=ax)
    out = salvar_fig(fig, "gap_eixos_2024_2025")
    plt.close(fig)
    return out


def fig_heatmap_segmento_questao(df: pd.DataFrame, eixos_alvo: list[int],
                                 nome_arquivo: str) -> Path:
    """Heatmap de médias por questão e segmento, em escala 1--6.

    Usa o colormap divergente RdYlGn centrado em ~3,5 (meio da escala) para
    facilitar a leitura: vermelho indica avaliação abaixo do meio, amarelo
    indica neutralidade, verde indica avaliação acima do meio.
    """
    questoes = []
    for e in eixos_alvo:
        questoes.extend(EIXOS_2025[e]["likert"])
    matriz = []
    for q in questoes:
        col = f"{q}_num"
        if col not in df.columns:
            matriz.append([np.nan] * len(SEGMENTOS_ORDEM))
            continue
        linha = []
        for seg in SEGMENTOS_ORDEM:
            v = df.loc[df["segmento"] == seg, col].dropna()
            linha.append(float(v.mean()) if len(v) > 0 else np.nan)
        matriz.append(linha)
    M = np.array(matriz)
    fig, ax = plt.subplots(figsize=(6.0, max(3.5, 0.30 * len(questoes))))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="RdYlGn", vmin=2, vmax=5.5,
                xticklabels=SEGMENTOS_ORDEM, yticklabels=questoes,
                cbar_kws={"label": "Média (1--6)", "shrink": 0.7},
                linewidths=0.4, linecolor="white",
                annot_kws={"fontsize": 8}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Questão")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    nome_sem_ext = nome_arquivo.replace(".png", "")
    out = salvar_fig(fig, nome_sem_ext)
    plt.close(fig)
    return out


def fig_heatmap_spearman(rho: pd.DataFrame) -> Path:
    """Heatmap das correlações de Spearman entre os escores compostos dos eixos.

    Versão diagnóstica/auxiliar: usada para verificar visualmente que as
    quatro dimensões SINAES estão associadas entre si na percepção dos
    respondentes. A análise substantiva é feita pelo heatmap
    questão-a-questão (`fig_heatmap_spearman_questoes`).
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(rho, annot=True, fmt=".3f", cmap="vlag", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.6, linecolor="white",
                cbar_kws={"label": r"$\rho$ de Spearman", "shrink": 0.75},
                annot_kws={"fontsize": 10}, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    out = salvar_fig(fig, "heatmap_spearman_eixos_2025")
    plt.close(fig)
    return out


def fig_heatmap_spearman_questoes(rho_q: pd.DataFrame) -> Path:
    """Heatmap das correlações de Spearman entre todas as questões universais.

    Matriz $k \\times k$ com células coloridas pelo coeficiente $\\rho$
    (azul para correlação positiva, branco para nula, vermelho para
    negativa). As células são anotadas com o valor de $\\rho$ apenas quando
    o coeficiente é forte ($|\\rho| \\geq 0{,}45$), para evitar poluição
    visual numa matriz grande. A diagonal principal é mascarada.
    """
    if rho_q.empty:
        return FIG_DIR / "heatmap_spearman_questoes_2025.png"
    n = rho_q.shape[0]
    # Mascara a diagonal para não poluir
    mask = np.eye(n, dtype=bool)
    # Anotações: só pares fortes (≥ 0,45 em módulo)
    anotacoes = rho_q.copy()
    anotacoes = anotacoes.where(rho_q.abs() >= 0.45, other="")
    anotacoes = anotacoes.applymap(
        lambda v: f"{float(v):.2f}" if isinstance(v, (int, float)) and not pd.isna(v) and v != "" else ""
    )
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(rho_q, mask=mask, cmap="vlag", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, linecolor="white",
                annot=anotacoes, fmt="",
                cbar_kws={"label": r"$\rho$ de Spearman", "shrink": 0.65},
                annot_kws={"fontsize": 6.5}, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    out = salvar_fig(fig, "heatmap_spearman_questoes_2025")
    plt.close(fig)
    return out


def fig_exemplo_boxplot_didatico(df: pd.DataFrame, questao: str = "q50") -> Path:
    """Boxplot didático anotado para a Fundamentação Teórica.

    Mostra a distribuição de uma questão Likert da CPA com anotações
    explícitas de mínimo, Q1, mediana, Q3, máximo, IQR e valores atípicos.
    Função pedagógica: serve de exemplo de leitura de boxplot na seção
    'Medidas de tendência central e dispersão' do capítulo de Fundamentação.
    """
    col = f"{questao}_num"
    if col not in df.columns:
        return FIG_DIR / "fig_exemplo_boxplot_didatico_2025.png"
    s = df[col].dropna()
    if s.empty:
        return FIG_DIR / "fig_exemplo_boxplot_didatico_2025.png"

    q1_v, mediana, q3_v = float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))
    iqr = q3_v - q1_v
    lim_inf, lim_sup = q1_v - 1.5 * iqr, q3_v + 1.5 * iqr
    minimo = float(max(s.min(), lim_inf))
    maximo = float(min(s.max(), lim_sup))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    cor_box = OKABE_ITO[1]
    cor_borda = OKABE_ITO[4]
    cor_mediana = OKABE_ITO[5]

    bp = ax.boxplot(
        [s.values], vert=False, widths=0.55, patch_artist=True,
        flierprops=dict(marker="o", markersize=5, markerfacecolor=cor_borda,
                        markeredgecolor=cor_borda, alpha=0.45),
        positions=[0],
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(cor_box); patch.set_edgecolor(cor_borda); patch.set_linewidth(1.2)
        patch.set_alpha(0.7)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(cor_borda); w.set_linewidth(1.0)
    for m in bp["medians"]:
        m.set_color(cor_mediana); m.set_linewidth(2.0)

    # Anotações explicativas
    ax.annotate(f"Q1 = {q1_v:.1f}", xy=(q1_v, 0.30), xytext=(q1_v - 0.4, 0.85),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7))
    ax.annotate(f"Mediana = {mediana:.1f}", xy=(mediana, 0.30), xytext=(mediana, 0.95),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=cor_mediana, lw=0.8))
    ax.annotate(f"Q3 = {q3_v:.1f}", xy=(q3_v, 0.30), xytext=(q3_v + 0.4, 0.85),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7))
    ax.annotate(f"mín. = {minimo:.1f}", xy=(minimo, 0), xytext=(minimo, -0.85),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7))
    ax.annotate(f"máx. = {maximo:.1f}", xy=(maximo, 0), xytext=(maximo, -0.85),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7))

    # IQR como faixa de fundo destacada
    ax.annotate("", xy=(q1_v, 0.55), xytext=(q3_v, 0.55),
                arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.8))
    ax.text((q1_v + q3_v) / 2, 0.62, f"IQR = {iqr:.1f}",
            ha="center", fontsize=9, color="0.3")

    # outliers (se houver na visualização do boxplot)
    out_low = s[s < lim_inf]
    out_high = s[s > lim_sup]
    if len(out_low) + len(out_high) > 0:
        ax.text(0.02, 0.95, f"Valores atípicos: {len(out_low) + len(out_high)} respondentes",
                transform=ax.transAxes, fontsize=8, color="0.4",
                verticalalignment="top")

    ax.set_yticks([])
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlabel("Resposta na escala Likert (1--6)")
    ax.set_ylim(-1.2, 1.3)
    sns.despine(ax=ax, left=True)
    out = salvar_fig(fig, "fig_exemplo_boxplot_didatico_2025")
    plt.close(fig)
    return out


def fig_exemplo_histograma(df: pd.DataFrame, eixo: int = 5) -> Path:
    """Histograma didático da distribuição do escore composto de um eixo.

    Função pedagógica: serve de exemplo de leitura de histograma na seção
    'Representação e Comunicação' do capítulo de Fundamentação. Usa o
    escore composto do eixo informado (default: Eixo 5 -- Infraestrutura).
    """
    col = f"score_eixo{eixo}"
    if col not in df.columns:
        return FIG_DIR / "fig_exemplo_histograma_2025.png"
    s = df[col].dropna()
    if s.empty:
        return FIG_DIR / "fig_exemplo_histograma_2025.png"

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cor_barra = OKABE_ITO[4]

    n, bins, patches = ax.hist(
        s.values, bins=20, color=cor_barra, alpha=0.85,
        edgecolor="white", linewidth=0.6,
    )

    media = float(s.mean())
    mediana = float(s.median())
    ax.axvline(media, color=OKABE_ITO[5], lw=1.6, ls="--",
               label=f"Média = {media:.2f}")
    ax.axvline(mediana, color=OKABE_ITO[2], lw=1.6, ls=":",
               label=f"Mediana = {mediana:.2f}")

    ax.set_xlabel(f"Escore composto do Eixo {eixo} (escala 1--6)")
    ax.set_ylabel("Número de respondentes")
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", color="0.93", linewidth=0.5, zorder=0)
    sns.despine(ax=ax)
    out = salvar_fig(fig, "fig_exemplo_histograma_2025")
    plt.close(fig)
    return out


def fig_exemplo_heatmap(df: pd.DataFrame, q1: str = "q34", q2: str = "q35") -> Path:
    """Heatmap didático de frequências cruzadas entre duas questões Likert.

    Função pedagógica: ilustra o conceito de heatmap como visualização de
    frequência conjunta de duas variáveis. Usa duas questões com correlação
    alta (default: q34 transparência da gestão x q35 divulgação de
    indicadores) para mostrar visualmente a estrutura ``diagonal'' que
    correlação positiva produz.
    """
    col1, col2 = f"{q1}_num", f"{q2}_num"
    if col1 not in df.columns or col2 not in df.columns:
        return FIG_DIR / "fig_exemplo_heatmap_2025.png"
    sub = df[[col1, col2]].dropna()
    if sub.empty:
        return FIG_DIR / "fig_exemplo_heatmap_2025.png"

    # Tabela 6x6 de frequências
    matriz = np.zeros((6, 6), dtype=int)
    for x, y in zip(sub[col1].astype(int), sub[col2].astype(int)):
        if 1 <= x <= 6 and 1 <= y <= 6:
            matriz[y - 1, x - 1] += 1

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Blues",
        cbar_kws={"label": "Número de respondentes", "shrink": 0.7},
        linewidths=0.4, linecolor="white",
        xticklabels=[1, 2, 3, 4, 5, 6],
        yticklabels=[1, 2, 3, 4, 5, 6],
        annot_kws={"fontsize": 9}, ax=ax,
    )
    ax.set_xlabel(f"{q1}")
    ax.set_ylabel(f"{q2}")
    ax.invert_yaxis()
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    out = salvar_fig(fig, "fig_exemplo_heatmap_2025")
    plt.close(fig)
    return out


def fig_par_estratificado(df: pd.DataFrame, q1: str, q2: str,
                    rho_agregado: float, rho_por_segmento: dict[str, float],
                    nome_arquivo: str, descricao_q1: str = "", descricao_q2: str = "") -> Path:
    """Comparação estratificada por segmento para um par de questões Likert.

    Para o par (q1, q2), gera um scatter plot dos respondentes em duas dimensões
    (q1 no eixo X, q2 no eixo Y), com pontos coloridos pelo segmento e jitter
    pequeno para visualizar a sobreposição (Likert é discreta, então sem jitter
    os pontos colapsam numa grade 6x6 ilegível). Sobre cada segmento é desenhada
    uma linha de tendência (regressão linear simples). No topo aparecem os
    coeficientes $\\rho$ de Spearman do agregado e de cada segmento, permitindo
    comparar a associação entre as duas questões quando medida na base inteira
    e quando medida dentro de cada segmento institucional separadamente.
    """
    col1, col2 = f"{q1}_num", f"{q2}_num"
    if col1 not in df.columns or col2 not in df.columns:
        return FIG_DIR / f"{nome_arquivo}.png"
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for seg in SEGMENTOS_ORDEM:
        if seg not in rho_por_segmento and seg != "Egresso":
            continue
        sub = df[(df["segmento"] == seg) & df[col1].notna() & df[col2].notna()]
        if len(sub) < 5:
            continue
        x = sub[col1].values + rng.uniform(-0.18, 0.18, size=len(sub))
        y = sub[col2].values + rng.uniform(-0.18, 0.18, size=len(sub))
        cor = SEG_COR.get(seg, "#888888")
        ax.scatter(x, y, s=18, color=cor, alpha=0.35, edgecolors="none",
                   label=f"{seg} (n={len(sub)})", zorder=2)
        # Linha de tendência
        if len(sub) >= 5:
            try:
                coef = np.polyfit(sub[col1].values, sub[col2].values, 1)
                xs = np.array([1, 6])
                ys = coef[0] * xs + coef[1]
                ax.plot(xs, ys, color=cor, lw=1.6, zorder=3, alpha=0.9)
            except (np.linalg.LinAlgError, ValueError):
                pass

    # Caixa de texto com os rhos por segmento
    linhas_rho = [f"$\\rho$ agregado: {rho_agregado:+.3f}"]
    for seg in SEGMENTOS_ORDEM:
        if seg in rho_por_segmento and pd.notna(rho_por_segmento[seg]):
            linhas_rho.append(f"$\\rho$ {seg.lower()}: {rho_por_segmento[seg]:+.3f}")
    texto = "\n".join(linhas_rho)
    ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.7", alpha=0.92))

    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.5, 6.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    label_x = f"{q1}" + (f" -- {descricao_q1}" if descricao_q1 else "")
    label_y = f"{q2}" + (f" -- {descricao_q2}" if descricao_q2 else "")
    ax.set_xlabel(label_x, fontsize=10)
    ax.set_ylabel(label_y, fontsize=10)
    ax.grid(color="0.93", linewidth=0.5, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    sns.despine(ax=ax)
    out = salvar_fig(fig, nome_arquivo)
    plt.close(fig)
    return out


def fig_nuvem_palavras(top_termos: pd.DataFrame) -> Path:
    """Nuvem de palavras dos comentários da q70 — apoio visual da análise textual."""
    if top_termos.empty:
        return FIG_DIR / "nuvem_q70_2025.png"
    freq = dict(zip(top_termos["termo"], top_termos["freq"]))
    wc = WordCloud(width=1200, height=600, background_color="white",
                   colormap="viridis", prefer_horizontal=0.92,
                   relative_scaling=0.45, min_font_size=10,
                   max_font_size=160, margin=4)
    wc.generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    out = salvar_fig(fig, "nuvem_q70_2025")
    plt.close(fig)
    return out


# ==============================================================================
# Captions / mapa de figuras
# ==============================================================================
#
# As figuras geradas pelo pipeline NÃO carregam título embutido. O título de
# cada figura aparece exclusivamente como caption no LaTeX --- a duplicação
# entre título embutido e caption tipográfica é considerada má prática em
# publicação científica. Este dicionário serve como referência rápida (``mapa''):
# para cada arquivo de figura, registra a caption recomendada e uma descrição
# do que a figura entrega. O pipeline também grava esse mapa como JSON em
# ``figuras/2025/figuras_captions.json`` para consulta externa.

import json

FIGURAS_DESCRICAO: dict[str, dict[str, str]] = {
    "media_ic_eixos_2025": {
        "caption": "Médias dos escores compostos por eixo SINAES, com intervalo de confiança 95\\% obtido por bootstrap percentílico (10\\,000 reamostragens) --- CPA/UFT 2025.",
        "descricao": "Pontos com barras de erro mostrando a média e o IC 95\\% bootstrap de cada eixo Likert (Eixos 2 a 5) no ciclo 2025.",
    },
    "boxplot_eixos_2025": {
        "caption": "Distribuição do percentual de satisfação por respondente em cada eixo SINAES, com pontos individuais sobrepostos --- CPA/UFT 2025. O número de itens que compõem cada eixo aparece entre parênteses no rótulo: a granularidade da métrica é $20/k$ pontos percentuais (onde $k$ é o número de itens), o que explica o aspecto ``em listras'' visível especialmente no Eixo 2 (5 itens, degraus de 4\\%).",
        "descricao": "Boxplot por eixo, com a métrica transformada de escala Likert 1--6 para porcentagem de satisfação 0--100 por respondente. Cada ponto representa um respondente; os pontos são desenhados à frente da caixa para mostrar simultaneamente a distribuição resumida e a dispersão real das observações. A granularidade depende do número de itens k de cada eixo (20/k pontos percentuais entre valores possíveis), o que explica as listras horizontais perceptíveis em eixos com poucos itens.",
    },
    "radar_segmentos_2025": {
        "caption": "Perfil de avaliação dos quatro segmentos institucionais (Discente, Docente, Técnico, Egresso) nos eixos SINAES --- CPA/UFT 2025.",
        "descricao": "Gráfico de radar com uma série por segmento; cada eixo do radar corresponde a um eixo SINAES e o raio é a média do escore composto naquele eixo para o segmento.",
    },
    "radar_2024_vs_2025": {
        "caption": "Comparação entre os ciclos 2024 e 2025 das médias dos escores compostos por eixo SINAES --- CPA/UFT.",
        "descricao": "Radar com duas séries sobrepostas (2024 e 2025), uma por ciclo, permitindo leitura visual imediata das variações por eixo.",
    },
    "gap_eixos_2024_2025": {
        "caption": "Variação 2024 → 2025 das médias por eixo SINAES, em colunas verticais agrupadas, com $\\Delta$ anotado --- CPA/UFT.",
        "descricao": "Para cada eixo, duas colunas lado a lado: 2024 (tom claro) e 2025 (tom escuro). A variação absoluta é anotada acima de cada par.",
    },
    "heatmap_segmento_eixos2_3_2025": {
        "caption": "Médias por questão e segmento institucional nos Eixos 2 (Desenvolvimento Institucional) e 3 (Políticas Acadêmicas) --- CPA/UFT 2025.",
        "descricao": "Heatmap divergente RdYlGn das médias 1--6 por questão (linhas) e segmento (colunas).",
    },
    "heatmap_segmento_eixos4_5_2025": {
        "caption": "Médias por questão e segmento institucional nos Eixos 4 (Políticas de Gestão) e 5 (Infraestrutura Física) --- CPA/UFT 2025.",
        "descricao": "Heatmap divergente RdYlGn das médias 1--6 por questão (linhas) e segmento (colunas).",
    },
    "heatmap_spearman_eixos_2025": {
        "caption": "Correlações de Spearman entre os escores compostos dos quatro eixos SINAES --- CPA/UFT 2025 (visão auxiliar).",
        "descricao": "Heatmap simétrico das correlações $\\rho$ de Spearman entre os quatro eixos Likert. Visão de validação interna do agrupamento por eixo: confirma que as quatro dimensões SINAES estão associadas entre si na percepção dos respondentes.",
    },
    "heatmap_spearman_questoes_2025": {
        "caption": "Matriz de correlações de Spearman entre todas as questões universais Likert do instrumento --- CPA/UFT 2025.",
        "descricao": "Heatmap $k \\times k$ com as correlações $\\rho$ de Spearman entre cada par de questões universais. Células anotadas apenas quando $|\\rho| \\geq 0{,}45$ para evitar poluição visual. Esta figura é mantida apenas como artefato técnico do pipeline (CSV em \\texttt{spearman\\_questoes\\_2025.csv}); não entra no corpo do artigo por ser ilegível em escala impressa. A análise substantiva é feita pelas tabelas auxiliares de top pares e robustez de Simpson.",
    },
    "fig_simpson_par_estavel_2025": {
        "caption": "Verificação do paradoxo de Simpson para um par com correlação estável entre segmentos: q34 (transparência da gestão) versus q35 (divulgação de indicadores) --- CPA/UFT 2025.",
        "descricao": "Scatter plot com jitter dos respondentes em duas dimensões Likert, coloridos pelo segmento institucional (Discente, Docente, Técnico, Egresso), com linhas de tendência por segmento. A caixa de texto indica os coeficientes $\\rho$ de Spearman no agregado e em cada segmento. Este par é exemplo de correlação que sobrevive ao teste de Simpson: $\\rho$ semelhante em todos os segmentos.",
    },
    "fig_simpson_par_heterogeneo_2025": {
        "caption": "Verificação do paradoxo de Simpson para um par com heterogeneidade entre segmentos: q30 (apoio psicopedagógico) versus q31 (atuação das instâncias superiores) --- CPA/UFT 2025.",
        "descricao": "Scatter plot com jitter dos respondentes em duas dimensões Likert, coloridos pelo segmento institucional, com linhas de tendência por segmento. A caixa de texto indica os coeficientes $\\rho$ de Spearman no agregado e em cada segmento. Este par exibe heterogeneidade entre os subgrupos: o achado contraintuitivo (o apoio psicopedagógico se acopla à governança alta) é forte em discentes e docentes, mas significativamente mais fraco em técnicos.",
    },
    "fig_significancia_kw_2025": {
        "caption": "Significância do teste de Kruskal-Wallis por questão universal, em escala $-\\log_{10}(p)$, com linhas verticais nos limiares críticos de $\\alpha=0{,}05$, Benjamini-Hochberg e Bonferroni --- CPA/UFT 2025.",
        "descricao": "Barras horizontais com $-\\log_{10}(p)$ por questão, ordenadas do mais significativo ao menos significativo. Quanto maior a barra, menor o $p$-valor. As três linhas verticais marcam os critérios de significância: a questão é significativa por um critério se sua barra ultrapassa a linha correspondente.",
    },
    "nuvem_q70_2025": {
        "caption": "Nuvem de palavras dos comentários da questão q70 (\\textit{a voz da comunidade}) --- CPA/UFT 2025.",
        "descricao": "Nuvem de palavras com tamanho proporcional à frequência dos termos após normalização e remoção de \\textit{stopwords}.",
    },
    "fig_exemplo_boxplot_didatico_2025": {
        "caption": "Exemplo de leitura de um boxplot, com Q1, mediana, Q3, IQR e bigodes anotados --- aplicado à questão q50 (Biblioteca do campus) da CPA/UFT 2025.",
        "descricao": "Boxplot horizontal didático com anotações explícitas dos cinco números da distribuição (mínimo, Q1, mediana, Q3, máximo) e do intervalo interquartil. Pedagógico: serve de exemplo de leitura de boxplot na Fundamentação Teórica.",
    },
    "fig_exemplo_histograma_2025": {
        "caption": "Exemplo de histograma da distribuição do escore composto do Eixo 5 (Infraestrutura Física) por respondente --- CPA/UFT 2025.",
        "descricao": "Histograma do escore composto do Eixo 5, com média e mediana sobrepostas como linhas verticais. Pedagógico: serve de exemplo de leitura de histograma na Fundamentação Teórica.",
    },
    "fig_exemplo_heatmap_2025": {
        "caption": "Exemplo de heatmap de frequências cruzadas entre duas questões Likert (q34 transparência da gestão $\\times$ q35 divulgação de indicadores) --- CPA/UFT 2025.",
        "descricao": "Heatmap $6 \\times 6$ da frequência conjunta de respostas em duas questões. Pedagógico: ilustra a estrutura visual diagonal que correlação positiva entre duas variáveis ordinais produz.",
    },
}


# ==============================================================================
# Parte 9 — Figuras para o kit relatório CPA
# ==============================================================================


def _carregar_enunciados() -> dict[str, str]:
    """Retorna dicionário q_id → texto da pergunta a partir do mapeamento."""
    mapa = carregar_mapeamento()
    return dict(zip(mapa["q_2025"], mapa["enunciado_2025"]))


def fig_tabela_campus_segmento(df: pd.DataFrame) -> Path:
    """Item 1: tabela visual campus × segmento, linhas zebradas, estilo CPA."""
    # Construir crosstab consolidado
    ct = pd.crosstab(
        df["segmento"], df["campus"].fillna("Sem campus"), margins=False
    )
    # Reordenar segmentos e campi
    seg_order = [s for s in SEG_PRIORIDADE if s in ct.index]
    campus_order = (
        ct.loc[seg_order].sum(axis=0).sort_values(ascending=False).index.tolist()
    )
    ct = ct.loc[seg_order, campus_order]
    # Adicionar totais
    ct.loc["Total por Segmento"] = ct.sum(axis=0)
    ct["Total"] = ct.sum(axis=1)
    total_participantes = int(ct.loc["Total por Segmento", "Total"])

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")

    n_rows, n_cols = ct.shape
    cell_text = ct.values.astype(int).astype(str).tolist()
    col_labels = list(ct.columns)
    row_labels = list(ct.index)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Cores
    header_color = "#0077A1"
    zebra_light = "#F5F5F5"
    zebra_white = "#FFFFFF"
    total_color = "#E8E8E8"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:  # header
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == n_rows:  # total row
            cell.set_facecolor(total_color)
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(zebra_light)
        else:
            cell.set_facecolor(zebra_white)
        # Row labels
        if col == -1:
            cell.set_text_props(fontweight="bold", ha="left")
            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color="white", fontweight="bold")
            elif row == n_rows:
                cell.set_facecolor(total_color)

    ax.set_title(
        f"Distribuição de respondentes por campus e segmento — CPA/UFT 2025\n"
        f"Total de participantes: {total_participantes}",
        fontsize=11, fontweight="bold", pad=12
    )
    fig.tight_layout()
    path = salvar_fig(fig, "tabela_campus_segmento_2025", subdir=KIT_DIR)
    plt.close(fig)
    return path


def fig_participacao_segmento_2024_2025(
    df_2025: pd.DataFrame, df_2024: pd.DataFrame
) -> Path:
    """Item 2: barras horizontais comparando respondentes absolutos por segmento, 2024 vs 2025."""
    segs = ["Docente", "Técnico", "Discente", "Egresso"]

    n_2024 = [(df_2024["segmento"] == seg).sum() for seg in segs]
    n_2025 = [(df_2025["segmento"] == seg).sum() for seg in segs]

    y = np.arange(len(segs))
    h = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars_24 = ax.barh(y + h / 2, n_2024, h, label="2024", color="#AAAAAA")
    bars_25 = ax.barh(y - h / 2, n_2025, h, label="2025", color="#0072B2")

    for bar, val in zip(bars_24, n_2024):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9)
    for bar, val in zip(bars_25, n_2025):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(segs, fontsize=10)
    ax.set_xlabel("Respondentes", fontsize=10)
    ax.set_ylabel("Segmento", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Respondentes por segmento — CPA/UFT 2024 vs 2025", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(max(n_2024), max(n_2025)) * 1.15)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = salvar_fig(fig, "participacao_segmento_2024_2025", subdir=KIT_DIR)
    plt.close(fig)
    return path


def _likert_subplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    questao: str,
    enunciados: dict[str, str],
    show_xticks: bool = False,
    is_first: bool = False,
) -> None:
    """Desenha um subplot Likert divergente por segmento numa Axes existente."""
    import textwrap

    cores_neg = ["#D32F2F", "#FF6F00", "#FBC02D"]
    cores_pos = ["#66BB6A", "#43A047", "#2E7D32"]

    col = questao if questao in df.columns else f"{questao}_num"
    if col not in df.columns:
        return

    segs = [s for s in reversed(SEGMENTOS_ORDEM) if (df["segmento"] == s).any()]

    rows = []
    for seg in segs:
        vals = df.loc[df["segmento"] == seg, col].dropna()
        n = len(vals)
        if n == 0:
            continue
        pcts = [(vals == v).sum() / n * 100 for v in range(1, 7)]
        rows.append({"seg": seg, "n": n, **{f"p{v}": pcts[v - 1] for v in range(1, 7)}})

    if not rows:
        return

    data = pd.DataFrame(rows)
    n_s = len(data)
    bar_h = 0.55
    y = np.arange(n_s) * 0.75

    left_3 = -data["p3"].values
    left_2 = left_3 - data["p2"].values

    ax.barh(y, -data["p3"], height=bar_h, left=0, color=cores_neg[2])
    ax.barh(y, -data["p2"], height=bar_h, left=left_3, color=cores_neg[1])
    ax.barh(y, -data["p1"], height=bar_h, left=left_2, color=cores_neg[0])
    ax.barh(y, data["p4"], height=bar_h, left=0, color=cores_pos[0])
    ax.barh(y, data["p5"], height=bar_h, left=data["p4"].values, color=cores_pos[1])
    ax.barh(y, data["p6"], height=bar_h, left=(data["p4"] + data["p5"]).values, color=cores_pos[2])

    # % dentro de cada pedaço
    min_pct = 3
    for i in range(n_s):
        rd = data.iloc[i]
        pieces = [
            (0, -rd["p3"], rd["p3"]),
            (left_3[i], -rd["p2"], rd["p2"]),
            (left_2[i], -rd["p1"], rd["p1"]),
            (0, rd["p4"], rd["p4"]),
            (rd["p4"], rd["p5"], rd["p5"]),
            (rd["p4"] + rd["p5"], rd["p6"], rd["p6"]),
        ]
        for left, width, pct in pieces:
            if pct >= min_pct:
                cx = left + width / 2
                ax.text(cx, y[i], f"{pct:.0f}%", va="center", ha="center",
                        fontsize=6, color="black")

    # Totais nas pontas
    for i in range(n_s):
        neg = data.iloc[i]["p1"] + data.iloc[i]["p2"] + data.iloc[i]["p3"]
        pos = data.iloc[i]["p4"] + data.iloc[i]["p5"] + data.iloc[i]["p6"]
        ax.text(-neg - 1.5, y[i], f"{neg:.0f}%", va="center", ha="right", fontsize=7)
        ax.text(pos + 1.5, y[i], f"{pos:.0f}%", va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{data.iloc[i]['seg']}s" for i in range(n_s)], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlim(-105, 105)

    # Eixo X: só no último subplot
    if show_xticks:
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_xticklabels(["100%", "50%", "0%", "50%", "100%"], fontsize=7)
        ax.set_xlabel("Frequência", fontsize=8)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    # Título = texto da pergunta, alinhado à esquerda
    titulo = enunciados.get(questao, questao)
    titulo_wrap = "\n".join(textwrap.wrap(titulo, width=95))
    ax.set_title(titulo_wrap, fontsize=10, fontweight="bold", loc="left", pad=3)

    # Box ao redor
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#AAAAAA")


def _gerar_painel_likert(
    df: pd.DataFrame,
    questoes: list[str],
    enunciados: dict[str, str],
    titulo_painel: str | None,
    nome_arquivo: str,
) -> Path:
    """Gera um painel de N questões empilhadas com legenda no rodapé."""
    from matplotlib.patches import Patch

    n_q = len(questoes)
    fig_h = n_q * 1.9 + 1.2
    fig, axes = plt.subplots(n_q, 1, figsize=(11, fig_h))
    if n_q == 1:
        axes = [axes]

    for idx, q in enumerate(questoes):
        is_last = (idx == n_q - 1)
        _likert_subplot(axes[idx], df, q, enunciados, show_xticks=is_last)

    cores_neg = ["#D32F2F", "#FF6F00", "#FBC02D"]
    cores_pos = ["#66BB6A", "#43A047", "#2E7D32"]
    labels = ["Péssimo", "Muito ruim", "Ruim", "Bom", "Muito bom", "Ótimo"]
    cores = cores_neg + cores_pos
    handles = [Patch(facecolor=c, label=l) for c, l in zip(cores, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
              frameon=True, fancybox=False, edgecolor="#AAAAAA",
              bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35, bottom=0.07)

    path = salvar_fig(fig, nome_arquivo, subdir=KIT_DIR)
    plt.close(fig)
    return path


def fig_likert_percentual_por_eixo(
    df: pd.DataFrame,
    eixo: int,
    enunciados: dict[str, str] | None = None,
) -> list[Path]:
    """Item 3: painéis de até 4 questões por página.

    Se o último grupo ficaria com apenas 1 questão sozinha,
    o penúltimo grupo absorve uma extra (fica com 5).
    """
    questoes = EIXOS_2025[eixo]["likert"]
    if not questoes:
        questoes = EIXOS_2025[eixo].get("condicionais", [])
    if not questoes:
        return []

    if enunciados is None:
        enunciados = _carregar_enunciados()

    # Dividir em grupos de 4, evitando grupo final com 1 sozinha
    max_por_grupo = 4
    grupos = []
    restante = list(questoes)
    while len(restante) > max_por_grupo:
        # Se sobrariam exatamente 1 no próximo corte, pegar 5 agora
        if len(restante) - max_por_grupo == 1:
            grupos.append(restante[:max_por_grupo + 1])
            restante = restante[max_por_grupo + 1:]
        else:
            grupos.append(restante[:max_por_grupo])
            restante = restante[max_por_grupo:]
    if restante:
        grupos.append(restante)

    paths = []
    for gi, grupo in enumerate(grupos):
        sufixo = f"_parte{gi + 1}" if len(grupos) > 1 else ""
        nome = f"likert_segmento_eixo{eixo}{sufixo}_2025"
        p = _gerar_painel_likert(df, grupo, enunciados, None, nome)
        if p:
            paths.append(p)

    return paths


def _likert_subplot_campus(
    ax: plt.Axes,
    df: pd.DataFrame,
    questao: str,
    enunciados: dict[str, str],
    campi: list[str],
    campus_n: dict[str, int],
    show_xticks: bool = False,
) -> None:
    """Desenha um subplot Likert divergente por campus numa Axes existente."""
    import textwrap

    cores_neg = ["#D32F2F", "#FF6F00", "#FBC02D"]
    cores_pos = ["#66BB6A", "#43A047", "#2E7D32"]

    col = questao if questao in df.columns else f"{questao}_num"
    if col not in df.columns:
        return

    rows = []
    for campus in campi:
        vals = df.loc[df["campus"] == campus, col].dropna()
        n = len(vals)
        if n == 0:
            pcts = [0] * 6
        else:
            pcts = [(vals == v).sum() / n * 100 for v in range(1, 7)]
        rows.append({"campus": campus, "n": n, **{f"p{v}": pcts[v - 1] for v in range(1, 7)}})

    data = pd.DataFrame(rows)
    n_c = len(data)
    bar_h = 0.55
    y = np.arange(n_c) * 0.75

    left_3 = -data["p3"].values
    left_2 = left_3 - data["p2"].values

    ax.barh(y, -data["p3"], height=bar_h, left=0, color=cores_neg[2])
    ax.barh(y, -data["p2"], height=bar_h, left=left_3, color=cores_neg[1])
    ax.barh(y, -data["p1"], height=bar_h, left=left_2, color=cores_neg[0])
    ax.barh(y, data["p4"], height=bar_h, left=0, color=cores_pos[0])
    ax.barh(y, data["p5"], height=bar_h, left=data["p4"].values, color=cores_pos[1])
    ax.barh(y, data["p6"], height=bar_h, left=(data["p4"] + data["p5"]).values, color=cores_pos[2])

    min_pct = 3
    for i in range(n_c):
        rd = data.iloc[i]
        pieces = [
            (0, -rd["p3"], rd["p3"]),
            (left_3[i], -rd["p2"], rd["p2"]),
            (left_2[i], -rd["p1"], rd["p1"]),
            (0, rd["p4"], rd["p4"]),
            (rd["p4"], rd["p5"], rd["p5"]),
            (rd["p4"] + rd["p5"], rd["p6"], rd["p6"]),
        ]
        for left, width, pct in pieces:
            if pct >= min_pct:
                cx = left + width / 2
                ax.text(cx, y[i], f"{pct:.0f}%", va="center", ha="center",
                        fontsize=6, color="black")

    for i in range(n_c):
        neg = data.iloc[i]["p1"] + data.iloc[i]["p2"] + data.iloc[i]["p3"]
        pos = data.iloc[i]["p4"] + data.iloc[i]["p5"] + data.iloc[i]["p6"]
        ax.text(-neg - 1.5, y[i], f"{neg:.0f}%", va="center", ha="right", fontsize=7)
        ax.text(pos + 1.5, y[i], f"{pos:.0f}%", va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([data.iloc[i]["campus"] for i in range(n_c)], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlim(-105, 105)

    if show_xticks:
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_xticklabels(["100%", "50%", "0%", "50%", "100%"], fontsize=7)
        ax.set_xlabel("Frequência", fontsize=8)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    titulo = enunciados.get(questao, questao)
    titulo_wrap = "\n".join(textwrap.wrap(titulo, width=95))
    ax.set_title(titulo_wrap, fontsize=10, fontweight="bold", loc="left", pad=3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#AAAAAA")


def fig_likert_percentual_por_campus(
    df: pd.DataFrame,
    eixo: int,
    enunciados: dict[str, str] | None = None,
    min_n: int = 30,
) -> list[Path]:
    """Item 4: painéis Likert por campus, até 4 questões por painel.

    Mesmo layout dos painéis por segmento, mas Y = campi em vez de segmentos.
    """
    from matplotlib.patches import Patch

    questoes = EIXOS_2025[eixo]["likert"]
    if not questoes:
        questoes = EIXOS_2025[eixo].get("condicionais", [])
    if not questoes:
        return []

    if enunciados is None:
        enunciados = _carregar_enunciados()

    # Campi válidos (N >= min_n), ordenados por média geral descendente
    campus_counts = df["campus"].value_counts()
    campi = sorted(
        [c for c in campus_counts.index if campus_counts[c] >= min_n],
        key=lambda c: df.loc[df["campus"] == c, f"score_eixo{eixo}"].mean()
        if f"score_eixo{eixo}" in df.columns else 0,
        reverse=True,
    )
    campus_n = {c: int(campus_counts[c]) for c in campi}

    if not campi:
        return []

    # Dividir em grupos de 4
    max_g = 4
    grupos = []
    restante = list(questoes)
    while len(restante) > max_g:
        if len(restante) - max_g == 1:
            grupos.append(restante[:max_g + 1])
            restante = restante[max_g + 1:]
        else:
            grupos.append(restante[:max_g])
            restante = restante[max_g:]
    if restante:
        grupos.append(restante)

    paths = []
    for gi, grupo in enumerate(grupos):
        n_q = len(grupo)
        fig_h = n_q * 1.9 + 1.2
        fig, axes = plt.subplots(n_q, 1, figsize=(11, fig_h))
        if n_q == 1:
            axes = [axes]

        for idx, q in enumerate(grupo):
            is_last = (idx == n_q - 1)
            _likert_subplot_campus(axes[idx], df, q, enunciados, campi, campus_n, show_xticks=is_last)

        cores_neg = ["#D32F2F", "#FF6F00", "#FBC02D"]
        cores_pos = ["#66BB6A", "#43A047", "#2E7D32"]
        labels = ["Péssimo", "Muito ruim", "Ruim", "Bom", "Muito bom", "Ótimo"]
        cores = cores_neg + cores_pos
        handles = [Patch(facecolor=c, label=l) for c, l in zip(cores, labels)]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
                  frameon=True, fancybox=False, edgecolor="#AAAAAA",
                  bbox_to_anchor=(0.5, -0.005))

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35, bottom=0.07)

        sufixo = f"_parte{gi + 1}" if len(grupos) > 1 else ""
        nome = f"likert_campus_eixo{eixo}{sufixo}_2025"
        p = salvar_fig(fig, nome, subdir=KIT_DIR)
        plt.close(fig)
        if p:
            paths.append(p)

    return paths


def fig_media_eixos_por_segmento(df: pd.DataFrame) -> Path:
    """Gráfico de linhas: média por eixo, uma linha por segmento + linha geral."""
    eixos_ids = [2, 3, 4, 5]
    eixos_nomes = [EIXOS_2025[e]["nome"] for e in eixos_ids]
    x_labels = [f"Eixo {e}\n{n}" for e, n in zip(eixos_ids, eixos_nomes)]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Linha geral (todos os respondentes)
    medias_geral = []
    for e in eixos_ids:
        col = f"score_eixo{e}"
        medias_geral.append(df[col].mean() if col in df.columns else np.nan)

    ax.plot(range(len(eixos_ids)), medias_geral, marker="o", markersize=8,
            linewidth=2.5, color="black", label="Geral", zorder=5)
    for i, v in enumerate(medias_geral):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    # Linha por segmento (sem egresso)
    for seg in ["Discente", "Docente", "Técnico"]:
        mask = df["segmento"] == seg
        medias = []
        for e in eixos_ids:
            col = f"score_eixo{e}"
            medias.append(df.loc[mask, col].mean() if col in df.columns else np.nan)
        cor = SEG_COR.get(seg, "#999999")
        ax.plot(range(len(eixos_ids)), medias, marker="s", markersize=6,
                linewidth=1.8, color=cor, label=f"{seg}s", alpha=0.85)
        for i, v in enumerate(medias):
            ax.text(i + 0.08, v - 0.1, f"{v:.2f}", ha="left", fontsize=7.5, color=cor)

    ax.set_xticks(range(len(eixos_ids)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Média (escala 1–6)", fontsize=10)
    ax.set_ylim(2.5, 5.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_title("Média por eixo SINAES e segmento — CPA/UFT 2025",
                 fontsize=12, fontweight="bold", pad=12)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = salvar_fig(fig, "media_eixos_por_segmento_2025", subdir=KIT_DIR)
    plt.close(fig)
    return path


def fig_media_eixos_por_campus(df: pd.DataFrame, min_n: int = 1) -> Path:
    """Gráfico de linhas: média por eixo, uma linha por campus + linha geral."""
    eixos_ids = [2, 3, 4, 5]
    eixos_nomes = [EIXOS_2025[e]["nome"] for e in eixos_ids]
    x_labels = [f"Eixo {e}\n{n}" for e, n in zip(eixos_ids, eixos_nomes)]

    campus_counts = df["campus"].value_counts()
    campi = [c for c in campus_counts.index if campus_counts[c] >= min_n and pd.notna(c)]
    # Ordenar por média geral descendente
    campi = sorted(campi, key=lambda c: df.loc[df["campus"] == c, "score_eixo2"].mean()
                   if "score_eixo2" in df.columns else 0, reverse=True)

    # Paleta para campi
    campus_cores = {}
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    for i, c in enumerate(campi):
        campus_cores[c] = palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Linha geral
    medias_geral = []
    for e in eixos_ids:
        col = f"score_eixo{e}"
        medias_geral.append(df[col].mean() if col in df.columns else np.nan)

    ax.plot(range(len(eixos_ids)), medias_geral, marker="o", markersize=8,
            linewidth=2.5, color="black", label="Geral", zorder=5)
    for i, v in enumerate(medias_geral):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    # Linha por campus
    for campus in campi:
        mask = df["campus"] == campus
        n_campus = int(mask.sum())
        medias = []
        for e in eixos_ids:
            col = f"score_eixo{e}"
            medias.append(df.loc[mask, col].mean() if col in df.columns else np.nan)
        cor = campus_cores[campus]
        ax.plot(range(len(eixos_ids)), medias, marker="s", markersize=5,
                linewidth=1.5, color=cor, label=f"{campus} (N={n_campus})", alpha=0.8)

    ax.set_xticks(range(len(eixos_ids)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Média (escala 1–6)", fontsize=10)
    ax.set_ylim(2.0, 5.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Média por eixo SINAES e campus — CPA/UFT 2025",
                 fontsize=12, fontweight="bold", pad=12)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = salvar_fig(fig, "media_eixos_por_campus_2025", subdir=KIT_DIR)
    plt.close(fig)
    return path


def _pie_data(series: pd.Series) -> tuple[list[float], list[str], list[str]]:
    """Calcula proporções Sim/Não/NSO a partir de uma coluna bruta (s/n/nso)."""
    s = series.dropna().astype(str).str.strip().str.lower()
    total = len(s)
    if total == 0:
        return [], [], []
    n_sim = (s == "s").sum()
    n_nao = (s == "n").sum()
    n_nso = (s == "nso").sum()
    sizes = [n_nao / total * 100, n_sim / total * 100, n_nso / total * 100]
    labels = [f"{sizes[0]:.0f}%", f"{sizes[1]:.0f}%", f"{sizes[2]:.0f}%"]
    nomes = ["Não", "Sim", "Não soube opinar"]
    # Filtrar fatias com 0%
    filtered = [(sz, lb, nm) for sz, lb, nm in zip(sizes, labels, nomes) if sz > 0]
    if not filtered:
        return [], [], []
    sizes, labels, nomes = zip(*filtered)
    return list(sizes), list(labels), list(nomes)


def fig_setores_binaria(
    df: pd.DataFrame,
    questao: str,
    enunciados: dict[str, str] | None = None,
) -> Path:
    """Gráfico de setores para questão binária: pizza geral + donuts por segmento.

    Se a questão é exclusiva de um segmento (ex: q6 só docentes), gera apenas
    um gráfico de pizza sem a divisão por segmento.
    """
    import textwrap

    if enunciados is None:
        enunciados = _carregar_enunciados()

    cores_pie = ["#336699", "#C8B200", "#8DB600"]  # Não, Sim, NSO

    # Determinar quais segmentos têm dados para esta questão
    col = questao
    if col not in df.columns:
        return Path()

    segs_com_dados = []
    for seg in SEGMENTOS_ORDEM:
        if seg == "Egresso":
            continue
        mask = df["segmento"] == seg
        vals = df.loc[mask, col].dropna()
        if len(vals) >= 5:
            segs_com_dados.append(seg)

    if not segs_com_dados:
        return Path()

    titulo = enunciados.get(questao, questao)
    titulo_wrap = "\n".join(textwrap.wrap(titulo, width=50))

    # Questão exclusiva de um segmento → pizza simples grande
    if len(segs_com_dados) == 1:
        seg = segs_com_dados[0]
        mask = df["segmento"] == seg
        sizes, labels, nomes = _pie_data(df.loc[mask, col])
        if not sizes:
            return Path()

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = [cores_pie[["Não", "Sim", "Não soube opinar"].index(n)] for n in nomes]
        wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                               startangle=90, textprops={"fontsize": 14, "fontweight": "bold"})
        ax.set_title(titulo_wrap, fontsize=15, fontweight="bold", pad=18)
        fig.legend(nomes, loc="upper right", fontsize=13, bbox_to_anchor=(0.98, 0.90),
                   framealpha=0.9, edgecolor="#CCCCCC")

        fig.tight_layout()
        path = salvar_fig(fig, f"setores_{questao}_2025", subdir=KIT_DIR)
        plt.close(fig)
        return path

    # Questão universal → pizza geral grande + donuts por segmento maiores
    n_segs = len(segs_com_dados)
    fig = plt.figure(figsize=(12, 9))

    # Pizza principal (centro-topo, maior)
    ax_main = fig.add_axes([0.2, 0.4, 0.6, 0.55])
    sizes_g, labels_g, nomes_g = _pie_data(df[col])
    if not sizes_g:
        plt.close(fig)
        return Path()
    colors_g = [cores_pie[["Não", "Sim", "Não soube opinar"].index(n)] for n in nomes_g]
    wedges, texts = ax_main.pie(sizes_g, labels=labels_g, colors=colors_g,
                                startangle=90, textprops={"fontsize": 14, "fontweight": "bold"})
    ax_main.set_title(titulo_wrap, fontsize=15, fontweight="bold", pad=15)
    fig.legend(nomes_g, loc="upper right", fontsize=13, bbox_to_anchor=(0.98, 0.96),
               framealpha=0.9, edgecolor="#CCCCCC")

    # Donuts por segmento (linha inferior, maiores)
    donut_size = 0.28
    total_width = donut_size * n_segs + 0.05 * (n_segs - 1)
    margin = (1.0 - total_width) / 2
    for si, seg in enumerate(segs_com_dados):
        x0 = margin + si * (donut_size + 0.05)
        ax_s = fig.add_axes([x0, 0.03, donut_size, 0.35])
        mask = df["segmento"] == seg
        sizes_s, labels_s, nomes_s = _pie_data(df.loc[mask, col])
        if not sizes_s:
            ax_s.axis("off")
            continue
        colors_s = [cores_pie[["Não", "Sim", "Não soube opinar"].index(n)] for n in nomes_s]
        wedges_s, texts_s = ax_s.pie(
            sizes_s, labels=labels_s, colors=colors_s,
            startangle=90, textprops={"fontsize": 11, "fontweight": "bold"},
            wedgeprops={"width": 0.5}
        )
        ax_s.set_title(f"{seg}s", fontsize=12, fontweight="bold", pad=8)


    path = salvar_fig(fig, f"setores_{questao}_2025", subdir=KIT_DIR)
    plt.close(fig)
    return path


def salvar_captions_json() -> Path:
    """Grava o mapa de captions/descrições em JSON na pasta de figuras."""
    out = FIG_DIR / "figuras_captions.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(FIGURAS_DESCRICAO, f, ensure_ascii=False, indent=2)
    return out


# ==============================================================================
# Pipeline em estágios — pontos de quebra para uso a partir do notebook
# ==============================================================================
#
# Cada estágio é uma função autônoma, com inputs/outputs bem definidos e prints
# de diagnóstico próprios. O notebook chama um estágio por célula, inspeciona o
# retorno e passa adiante. ``pipeline_2025()``, no fim deste bloco, encadeia
# todos os estágios na ordem canônica e é a entrada usada quando o arquivo é
# executado diretamente (``python analise_dados_2025.py``).
#
# Ordem canônica:
#   1. carregar_e_consolidar()  → df
#   2. descritiva(df)           → dict
#   3. inferencial(df)          → dict (contém o mapeamento 2024↔2025)
#   4. textual(df)              → dict
#   5. temporal(df, mapa)       → dict (carrega 2024 e calcula variações)
#   6. figuras(df, ctx)         → list[Path]
# ==============================================================================

def carregar_e_consolidar(path: Path = PATH_2025, ano: int = 2025) -> pd.DataFrame:
    """Estágio 1 — Carregamento e consolidação.

    Lê a base bruta, aplica a regra de prioridade ``Doc > Téc > Disc > Egr``
    (Hurlbert, 1984), determina o campus a partir da coluna do segmento
    consolidado, normaliza colunas Likert/binárias/condicionais e adiciona os
    scores compostos por eixo. Retorna o DataFrame consolidado pronto para
    análise.
    """
    section(f"Estágio 1 — Carregamento e consolidação ({ano})")
    df_raw = carregar_base(path, ano=ano)
    print(f"  Linhas brutas: {df_raw.shape[0]} | Colunas: {df_raw.shape[1]}")

    df = atribuir_segmento_unico(df_raw)
    df = atribuir_campus(df)

    n_antes = len(df)
    df = df[df["segmento"].notna()].reset_index(drop=True)
    print(f"  Após filtro de segmento válido: {len(df)} (descartados: {n_antes - len(df)})")

    todas_likert = listar_questoes_likert()
    todas_bin    = listar_questoes_binarias()
    todas_cond   = listar_questoes_condicionais()

    df = normalizar_likert(df, todas_likert + todas_cond)
    df = normalizar_binarias(df, todas_bin)
    df = adicionar_scores_compostos(df)

    print("\n  Distribuição por segmento (após regra de prioridade):")
    print(df["segmento"].value_counts().to_string())
    return df


def descritiva(df: pd.DataFrame, salvar: bool = True) -> dict:
    """Estágio 2 — Estatística descritiva.

    Calcula: descritiva por questão, descritiva por eixo, taxa de NSO (total e
    por segmento), descritiva por campus, descritiva por campus×segmento e
    descritiva por curso (discentes, ``min_n=10``). Se ``salvar=True``, escreve
    cada DataFrame na pasta ``outputs/2025/``.

    Retorna ``dict`` com as chaves: ``por_questao``, ``por_eixo``, ``nso``,
    ``nso_por_segmento``, ``por_campus``, ``por_campus_segmento``, ``por_curso``.
    """
    section("Estágio 2 — Estatística descritiva")
    todas_likert = listar_questoes_likert()

    desc_q          = descritiva_por_questao(df, todas_likert)
    desc_e          = descritiva_por_eixo(df)
    desc_nso        = taxa_nso_por_questao(df, todas_likert)
    desc_nso_seg    = taxa_nso_por_questao(df, todas_likert, segmento_col="segmento")
    desc_campus     = descritiva_por_campus(df)
    desc_campus_seg = descritiva_por_campus_segmento(df)
    desc_curso      = descritiva_por_curso_discentes(df, min_n=10)

    if salvar:
        desc_q.to_csv(OUT_DIR / "descritiva_por_questao_2025.csv", index=False, encoding="utf-8")
        desc_e.to_csv(OUT_DIR / "descritiva_por_eixo_2025.csv", index=False, encoding="utf-8")
        desc_nso.to_csv(OUT_DIR / "cobertura_nso_por_questao_2025.csv", index=False, encoding="utf-8")
        desc_nso_seg.to_csv(OUT_DIR / "cobertura_nso_por_segmento_2025.csv", index=False, encoding="utf-8")
        desc_campus.to_csv(OUT_DIR / "descritiva_por_campus_2025.csv", index=False, encoding="utf-8")
        desc_campus_seg.to_csv(OUT_DIR / "descritiva_por_campus_segmento_2025.csv", index=False, encoding="utf-8")
        if not desc_curso.empty:
            desc_curso.to_csv(OUT_DIR / "descritiva_por_curso_discentes_2025.csv", index=False, encoding="utf-8")

    print("  Top 10 questões por média:")
    print(desc_q.head(10)[["questao", "eixo", "N", "media", "dp"]].to_string(index=False))
    print("\n  Médias por eixo:")
    print(desc_e[["eixo", "nome", "media", "dp", "N"]].to_string(index=False))
    print("\n  Top 10 questões com maior taxa de NSO (não sei opinar):")
    print(desc_nso.head(10).to_string(index=False))
    print("\n  Médias por eixo e campus (N >= 30):")
    print(desc_campus.to_string(index=False))
    if not desc_curso.empty:
        print("\n  Médias por eixo e curso (discentes, N >= 10):")
        print(desc_curso.to_string(index=False))

    return {
        "por_questao": desc_q,
        "por_eixo": desc_e,
        "nso": desc_nso,
        "nso_por_segmento": desc_nso_seg,
        "por_campus": desc_campus,
        "por_campus_segmento": desc_campus_seg,
        "por_curso": desc_curso,
    }


def inferencial(df: pd.DataFrame, salvar: bool = True) -> dict:
    """Estágio 3 — Correlações e consistência interna.

    Carrega o mapeamento 2024↔2025, identifica as questões universais
    (presentes para os 4 segmentos), calcula a matriz Spearman inter-eixos, a
    matriz Spearman questão-a-questão (agregada e estratificada por segmento),
    os top pares e a tabela de estratificação, e o alfa de Cronbach por eixo.

    Retorna ``dict`` com as chaves: ``mapeamento``, ``questoes_universais``,
    ``spearman_eixos``, ``spearman_questoes``, ``spearman_questoes_estratificado``,
    ``top_pares``, ``estratificacao_por_segmento``, ``cronbach``.
    """
    section("Estágio 3 — Correlações e consistência interna")

    mapa = carregar_mapeamento()
    universais = determinar_questoes_universais(mapa)
    print(f"  Questões universais (presentes para os 4 segmentos): {len(universais)}")

    rho    = spearman_inter_eixos(df)
    rho_q  = spearman_questao_a_questao(df, universais)
    top_pares = top_pares_spearman(rho_q, n=30)
    rho_q_estratificado = spearman_questao_a_questao_estratificado(
        df, universais, segmentos=["Discente", "Docente", "Técnico"]
    )
    tab_estratificacao = tabela_estratificacao_por_segmento(rho_q, rho_q_estratificado, n=20)
    cron = cronbach_por_eixo(df)

    if salvar:
        rho.to_csv(OUT_DIR / "spearman_eixos_2025.csv", encoding="utf-8")
        rho_q.to_csv(OUT_DIR / "spearman_questoes_2025.csv", encoding="utf-8")
        top_pares.to_csv(OUT_DIR / "top_pares_spearman_2025.csv", index=False, encoding="utf-8")
        for seg, mat in rho_q_estratificado.items():
            seg_norm = seg.lower().replace("é", "e").replace("ê", "e")
            mat.to_csv(OUT_DIR / f"spearman_questoes_{seg_norm}_2025.csv", encoding="utf-8")
        tab_estratificacao.to_csv(OUT_DIR / "estratificacao_por_segmento_2025.csv", index=False, encoding="utf-8")
        cron.to_csv(OUT_DIR / "cronbach_eixos_2025.csv", index=False, encoding="utf-8")

    print("\n  Spearman entre eixos (auxiliar):")
    print(rho.to_string())
    print(f"\n  Spearman questão-a-questão: matriz {rho_q.shape[0]}x{rho_q.shape[1]}")
    print("  Top 15 pares por |rho|:")
    print(top_pares.head(15).to_string(index=False))
    print("\n  Estratificação por segmento — top 10 pares com rho por segmento:")
    print(tab_estratificacao.head(10).to_string(index=False))
    print("\n  Cronbach alpha por eixo:")
    print(cron.to_string(index=False))

    return {
        "mapeamento": mapa,
        "questoes_universais": universais,
        "spearman_eixos": rho,
        "spearman_questoes": rho_q,
        "spearman_questoes_estratificado": rho_q_estratificado,
        "top_pares": top_pares,
        "estratificacao_por_segmento": tab_estratificacao,
        "cronbach": cron,
    }


def textual(df: pd.DataFrame, salvar: bool = True) -> dict:
    """Estágio 4 — Análise textual da q70.

    Estatísticas básicas do corpus (volume, extensão), frequência de termos,
    bigramas e categorização por eixo SINAES via dicionário temático.

    Retorna ``dict`` com as chaves: ``estat``, ``top_termos``, ``top_bigramas``,
    ``categorizacao``.
    """
    section("Estágio 4 — Análise textual da q70")
    estat_q70, top_termos, top_bigramas, cat_q70 = analise_q70(df)

    if salvar:
        top_termos.to_csv(OUT_DIR / "freq_termos_q70_2025.csv", index=False, encoding="utf-8")
        cat_q70.to_csv(OUT_DIR / "categorizacao_q70_2025.csv", index=False, encoding="utf-8")
        pd.DataFrame([estat_q70]).to_csv(OUT_DIR / "estat_q70_2025.csv", index=False, encoding="utf-8")

    print(f"  Comentários: {estat_q70.get('com_comentario', 0)} de {estat_q70.get('total_respondentes', 0)} "
          f"({estat_q70.get('pct_com_comentario', 0)}%)")
    print(f"  Extensão média: {estat_q70.get('media_palavras', 0)} palavras")
    print("\n  Top 15 termos:")
    print(top_termos.head(15).to_string(index=False))
    print("\n  Categorização por eixo:")
    print(cat_q70.to_string(index=False))

    return {
        "estat": estat_q70,
        "top_termos": top_termos,
        "top_bigramas": top_bigramas,
        "categorizacao": cat_q70,
    }


def temporal(df_2025: pd.DataFrame, mapa: pd.DataFrame, salvar: bool = True) -> dict:
    """Estágio 5 — Comparação temporal 2024 → 2025.

    Carrega a base de 2024, aplica o mesmo pipeline de consolidação (com o
    catálogo de eixos reconstruído a partir do mapeamento) e calcula a tabela
    de variações por questão e as médias por eixo de cada ano.

    Requer o ``mapa`` produzido por :func:`inferencial`.

    Retorna ``dict`` com as chaves: ``df_2024``, ``comparacao``,
    ``medias_2024``, ``medias_2025``.
    """
    section("Estágio 5 — Comparação temporal 2024 → 2025")
    df_2024_raw = carregar_base(PATH_2024, ano=2024)
    df_2024 = atribuir_segmento_unico(df_2024_raw)
    df_2024 = atribuir_campus(df_2024)
    df_2024 = df_2024[df_2024["segmento"].notna()].reset_index(drop=True)

    # Em 2024 não temos q60/q61; o catálogo é reconstruído a partir do mapeamento.
    todas_likert_2024 = [m["q_2024"] for _, m in mapa.iterrows()
                         if m["tipo"] == "Likert" and m["q_2024"] not in (None, "", float("nan"))]
    df_2024 = normalizar_likert(df_2024, list(set(todas_likert_2024)))
    catalogo_2024 = _construir_catalogo_2024(mapa)
    df_2024 = adicionar_scores_compostos(df_2024, eixos=catalogo_2024)

    comparacao = comparacao_temporal(df_2024, df_2025, mapa)
    med_2024 = medias_eixo_por_ano(df_2024, eixos=catalogo_2024)
    med_2025 = medias_eixo_por_ano(df_2025)

    if salvar:
        comparacao.to_csv(OUT_DIR / "temporal_2024_vs_2025.csv", index=False, encoding="utf-8")

    print(f"  N (2024) após regra de prioridade: {len(df_2024)}")
    print(f"  Questões comparadas: {len(comparacao)}")
    if not comparacao.empty:
        print("\n  Top 10 maiores variações:")
        print(comparacao.head(10)[["q_2025", "q_2024", "eixo", "media_2024",
                                   "media_2025", "delta"]].to_string(index=False))

    return {
        "df_2024": df_2024,
        "comparacao": comparacao,
        "medias_2024": med_2024,
        "medias_2025": med_2025,
    }


def figuras(df: pd.DataFrame, ctx: dict) -> list[Path]:
    """Estágio 6 — Geração de figuras (TCC + kit relatório CPA).

    ``ctx`` é o conjunto de artefatos produzido pelos estágios anteriores e
    deve conter as chaves: ``desc_e``, ``rho``, ``rho_q``,
    ``rho_q_estratificado``, ``df_2024``, ``med_2024``, ``med_2025``,
    ``top_termos``, ``mapa``.

    Salva PNGs em ``figuras/2025/`` (e ``figuras/2025/kit_cpa/`` para o kit) e
    retorna a lista de :class:`Path` das figuras geradas.
    """
    section("Estágio 6 — Geração de figuras")

    desc_e              = ctx["desc_e"]
    rho                 = ctx["rho"]
    rho_q               = ctx["rho_q"]
    rho_q_estratificado = ctx["rho_q_estratificado"]
    df_2024             = ctx["df_2024"]
    med_2024            = ctx["med_2024"]
    med_2025            = ctx["med_2025"]
    top_termos          = ctx["top_termos"]
    mapa                = ctx["mapa"]

    figs: list[Path] = []
    figs.append(fig_media_por_eixo(desc_e))
    figs.append(fig_boxplot_por_eixo(df))
    figs.append(fig_radar_segmentos(df))
    figs.append(fig_radar_campus(df))

    # Heatmap questão × campus para as questões com maior gap Disc-Doc
    # (mesmas questões do achado principal da comparação por segmento).
    questoes_destaque = ["q53", "q23", "q16", "q39", "q42", "q27", "q37", "q15", "q26", "q25"]
    figs.append(fig_heatmap_questao_campus(df, questoes_destaque))
    figs.append(fig_radar_2024_2025(med_2024, med_2025))
    figs.append(fig_gap_eixos_2024_2025(med_2024, med_2025))
    figs.append(fig_heatmap_segmento_questao(df, [2, 3], "heatmap_segmento_eixos2_3_2025.png"))
    figs.append(fig_heatmap_segmento_questao(df, [4, 5], "heatmap_segmento_eixos4_5_2025.png"))
    figs.append(fig_heatmap_spearman(rho))

    # Duas figuras de pares estratificados por segmento:
    # (a) par estável: q34 (transparência) × q35 (indicadores), rho ~0,74
    # (b) par com heterogeneidade: q30 (apoio psicopedagógico) × q31 (instâncias
    #     superiores), rho ~0,73 no agregado mas variando entre segmentos.
    enunciados_q = dict(zip(mapa["q_2025"], mapa["enunciado_2025"]))
    def _curto(q):
        s = enunciados_q.get(q, "")
        return s[:55] + "…" if len(s) > 55 else s

    for q1_par, q2_par, nome_fig in [
        ("q34", "q35", "fig_par_estavel_2025"),
        ("q30", "q31", "fig_par_heterogeneo_2025"),
    ]:
        if q1_par in rho_q.index and q2_par in rho_q.columns:
            rho_ag = float(rho_q.loc[q1_par, q2_par])
            rho_seg = {}
            for seg, mat in rho_q_estratificado.items():
                if q1_par in mat.index and q2_par in mat.columns:
                    rho_seg[seg] = float(mat.loc[q1_par, q2_par])
            figs.append(
                fig_par_estratificado(
                    df, q1_par, q2_par, rho_ag, rho_seg,
                    nome_fig, _curto(q1_par), _curto(q2_par),
                )
            )

    figs.append(fig_nuvem_palavras(top_termos))

    # Figuras didáticas para a Fundamentação Teórica.
    figs.append(fig_exemplo_boxplot_didatico(df, questao="q50"))
    figs.append(fig_exemplo_histograma(df, eixo=5))
    figs.append(fig_exemplo_heatmap(df, q1="q34", q2="q35"))

    # ----- Kit relatório CPA -----
    print("\n  Kit relatório CPA — itens adicionais")
    figs.append(fig_tabela_campus_segmento(df))
    figs.append(fig_participacao_segmento_2024_2025(df, df_2024))
    enunciados_dict = _carregar_enunciados()
    for eixo_id in [1, 2, 3, 4, 5]:
        paths = fig_likert_percentual_por_eixo(df, eixo_id, enunciados_dict)
        figs.extend(paths)
    binarias_kit = ["q1", "q2", "q3", "q4", "q5", "q6", "q28", "q60", "q61"]
    for q in binarias_kit:
        if q in df.columns:
            p = fig_setores_binaria(df, q, enunciados_dict)
            if p:
                figs.append(p)
    figs.append(fig_media_eixos_por_segmento(df))
    figs.append(fig_media_eixos_por_campus(df))
    for eixo_id in [1, 2, 3, 4, 5]:
        paths = fig_likert_percentual_por_campus(df, eixo_id, enunciados_dict, min_n=1)
        figs.extend(paths)

    salvar_captions_json()
    print("\n  Figuras geradas:")
    for f in figs:
        print(f"    {f.name}")
    return figs


# ------------------------------------------------------------------------------
# Wrapper que encadeia os 6 estágios — entry point CLI
# ------------------------------------------------------------------------------

def pipeline_2025() -> dict:
    """Encadeia os seis estágios na ordem canônica e devolve um dicionário com
    todos os artefatos. Equivalente a, no notebook, chamar:

        df   = carregar_e_consolidar()
        desc = descritiva(df)
        inf  = inferencial(df)
        text = textual(df)
        tmp  = temporal(df, mapa=inf["mapeamento"])
        figs = figuras(df, ctx={...})
    """
    df   = carregar_e_consolidar()
    desc = descritiva(df)
    inf  = inferencial(df)
    text = textual(df)
    tmp  = temporal(df, mapa=inf["mapeamento"])

    ctx = {
        "desc_e": desc["por_eixo"],
        "rho": inf["spearman_eixos"],
        "rho_q": inf["spearman_questoes"],
        "rho_q_estratificado": inf["spearman_questoes_estratificado"],
        "df_2024": tmp["df_2024"],
        "med_2024": tmp["medias_2024"],
        "med_2025": tmp["medias_2025"],
        "top_termos": text["top_termos"],
        "mapa": inf["mapeamento"],
    }
    figs = figuras(df, ctx)

    section("Síntese final")
    print(f"  N (2025) após regra de prioridade: {int(len(df))}")
    print(f"  N (2024) após regra de prioridade: {int(len(tmp['df_2024']))}")
    print(f"  Distribuição 2025: {df['segmento'].value_counts().to_dict()}")
    print(f"  Distribuição 2024: {tmp['df_2024']['segmento'].value_counts().to_dict()}")

    return {
        "df_2025": df,
        "df_2024": tmp["df_2024"],
        "descritiva_questao": desc["por_questao"],
        "descritiva_eixo": desc["por_eixo"],
        "spearman_eixos": inf["spearman_eixos"],
        "spearman_questoes": inf["spearman_questoes"],
        "spearman_questoes_estratificado": inf["spearman_questoes_estratificado"],
        "top_pares_spearman": inf["top_pares"],
        "estratificacao_por_segmento": inf["estratificacao_por_segmento"],
        "cronbach": inf["cronbach"],
        "temporal": tmp["comparacao"],
        "q70_estat": text["estat"],
        "q70_top": text["top_termos"],
        "q70_categorizacao": text["categorizacao"],
        "med_eixos_2024": tmp["medias_2024"],
        "med_eixos_2025": tmp["medias_2025"],
        "figuras": figs,
    }


def _construir_catalogo_2024(mapa: pd.DataFrame) -> dict:
    """Reconstrói o catálogo de eixos do instrumento 2024 a partir do mapeamento.

    Permite calcular scores compostos do 2024 usando as questões na numeração de 2024.
    """
    cat = {}
    for eixo in [1, 2, 3, 4, 5]:
        sub = mapa[(mapa["eixo"] == str(eixo)) | (mapa["eixo"] == eixo)]
        likert = [m["q_2024"] for _, m in sub.iterrows()
                  if m["tipo"] == "Likert" and m["q_2024"] not in (None, "", float("nan"))]
        binarias = [m["q_2024"] for _, m in sub.iterrows()
                    if m["tipo"] == "Binaria" and m["q_2024"] not in (None, "", float("nan"))]
        condicionais = [m["q_2024"] for _, m in sub.iterrows()
                        if m["tipo"] == "Condicional" and m["q_2024"] not in (None, "", float("nan"))]
        cat[eixo] = {
            "nome": EIXOS_2025[eixo]["nome"],
            "binarias": binarias,
            "condicionais": condicionais,
            "likert": likert,
        }
    return cat


if __name__ == "__main__":
    try:
        artefatos = pipeline_2025()
        section("CONCLUÍDO")
        print(f"  CSVs em: {OUT_DIR}")
        print(f"  Figuras em: {FIG_DIR}")
    except Exception as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
