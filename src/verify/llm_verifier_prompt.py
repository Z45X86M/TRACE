"""Single source of truth for the LLM verifier prompt (3-way classification).

Mirrors docs/llm_verifier_prompt.md. llm_rerank_api.py imports from here so the
prompt the CODE runs is written out in exactly one place. (The verifier is now
API-only; the old local-model runners have been removed.)

PROMPTING TECHNIQUE: CoT-Pro (Co2FuLL, ASE'25 — best for smaller models). The
model is given explicit GUIDING QUESTIONS from a syntactic and a semantic
perspective, answers them briefly (the chain of thought), then INTEGRATES the
answers into exactly one of three labels. We do NOT ask for any numeric score
(unreliable on small models) — only the discrete label. The CoT instructions +
guiding questions + few-shot reasoning demos are all STATIC, so they live in the
first-level cached prefix together with the few-shot.

The final line is always `ANSWER: <LABEL>`; the runtime parses that (parse_label)
to get the discrete class. Reranking is then a label-driven stable promotion
(matched candidates moved to the front in Stage-B order) — done by the caller,
NOT by any LLM score.

Few-shot examples are real decompiled pseudocode pulled from partial_100K, one
per class (see the .md for the per-example rationale).
"""

LABELS = ["NO_MATCH", "HOST_MATCH", "INLINE_MATCH"]
MATCH_LABEL_IDX = [1, 2]

SYSTEM = (
    "You are an expert in binary reverse engineering. You are given two functions "
    "recovered by a decompiler. They may have been compiled for DIFFERENT CPU "
    "architectures and DIFFERENT optimization levels, so identical source code can "
    "look syntactically very different (renamed variables, helpers inlined or not, "
    "reordered branches). One function may also be only a FRAGMENT that was inlined "
    "into a larger host function.\n\n"
    "Your task: classify the relationship between Function A (the query) and "
    "Function B (the candidate) into exactly one of three labels:\n"
    "- NO_MATCH      : A and B come from DIFFERENT source functions. Output this "
    "even when they share a library, have similar names, or share one algorithmic "
    "snippet — judge by what the bodies actually compute, not by the names.\n"
    "- HOST_MATCH    : A and B are the SAME complete top-level function (one may "
    "have inlined a callee the other called normally, but the outer identity is "
    "the same).\n"
    "- INLINE_MATCH  : B is the standalone version of a callee that appears INLINED "
    "inside A (or A is an inlined fragment whose standalone form is B). The match is "
    "at the inlined-callee level, not the whole outer function.\n\n"
    "Reason step by step by answering these GUIDING QUESTIONS briefly (one short "
    "sentence each), then integrate them:\n"
    "Syntactic level:\n"
    "  S1. What distinctive constants, string literals and called-function names "
    "does each side use, and do they overlap?\n"
    "  S2. What is each side's control-flow skeleton (loops, branches, early "
    "returns), and do the skeletons align?\n"
    "Semantic level:\n"
    "  M1. In one sentence, what does each function compute (its core algorithm)?\n"
    "  M2. Is B the whole of A (HOST), or does B's logic appear as an inlined "
    "fragment inside a larger A / vice-versa (INLINE), or are they unrelated (NO)? "
    "A callee still counts as INLINE_MATCH even when it is SMALL or trivial and A is "
    "much larger or does much more — and even when A inlines it MORE THAN ONCE. What "
    "matters is that B's distinctive computation (its loop/constants/calls) appears "
    "as a fragment inside A, NOT the size difference; do not answer NO_MATCH just "
    "because B is a tiny helper.\n\n"
    "After the four answers, output a final line of EXACTLY the form:\n"
    "ANSWER: <one of NO_MATCH | HOST_MATCH | INLINE_MATCH>"
)

PAIR_TMPL = ("Function A (query):\n```c\n{q}\n```\n\n"
             "Function B (candidate):\n```c\n{c}\n```")

FEW_SHOT = [
    (
        "void *__fastcall sub_137544(const char *a1, int a2)\n"
        "  if ( !a1 )\n"
        "    v7 = 0;\n"
        "    *(_DWORD *)sub_135DD4() = 1;\n"
        "    return v7;\n"
        "  v4 = strlen(a1);\n"
        "  v5 = v4 + 1;\n"
        "  v6 = a2 == 0;\n"
        "  if ( a2 )\n"
        "    v6 = dword_20B3A0 == 0;\n"
        "  if ( v6 )\n"
        "    v7 = malloc(v4 + 1);\n"
        "    if ( v7 )\n"
        "      goto LABEL_6;\n"
        "  else\n"
        "    v7 = (void *)(*(int (__fastcall **)(size_t, int))dword_20B3A0)(v4 + 1, a2);\n"
        "    if ( v7 )\n"
        "LABEL_6:\n"
        "      memmove(v7, a1, v5);\n"
        "      return v7;\n"
        "  *(_DWORD *)sub_135DD4() = 2;\n"
        "  return v7;",
        "char *__cdecl ber_strdup_x(const char *s, void *ctx)\n"
        "  if ( !s )\n"
        "    *ber_errno_addr() = 1;\n"
        "    return 0;\n"
        "  v2 = strlen(s) + 1;\n"
        "  v3 = ber_memalloc_x(v2, ctx);\n"
        "  if ( !v3 )\n"
        "    return 0;\n"
        "  v4 = v3;\n"
        "  memmove(v3, s, v2);\n"
        "  return (char *)v4;",
        "S1. Both use strlen + memmove and set an errno-like global, but A picks its "
        "allocator from a global function-pointer hook (dword_20B3A0) or malloc, "
        "while B always calls ber_memalloc_x(size, ctx).\n"
        "S2. Skeletons rhyme (null-check, length+1, allocate, copy, return) but A "
        "has an extra hook-vs-malloc branch that B lacks.\n"
        "M1. A = strdup that selects the allocator from a global hook/malloc; "
        "B = strdup that always allocates from the passed-in context.\n"
        "M2. The allocation source differs, so they are unrelated source functions, "
        "not a whole-vs-inlined relationship.",
        "NO_MATCH",
    ),
    (
        "__int64 sqlite3_os_init()\n"
        "  sqlite3_vfs_register();\n"
        "  sqlite3_vfs_register();\n"
        "  sqlite3_vfs_register();\n"
        "  sqlite3_vfs_register();\n"
        "  qword_CF388 = sub_FF40();\n"
        "  qword_CF1F8 = (__int64)getenv(\"SQLITE_TMPDIR\");\n"
        "  qword_CF200 = (__int64)getenv(\"TMPDIR\");\n"
        "  return 0LL;",
        "int sqlite3_os_init()\n"
        "  sqlite3_vfs_register(aVfs_86, 1);\n"
        "  sqlite3_vfs_register(&aVfs_86[1], 0);\n"
        "  sqlite3_vfs_register(&aVfs_86[2], 0);\n"
        "  sqlite3_vfs_register(&aVfs_86[3], 0);\n"
        "  unixBigLock = sqlite3MutexAlloc(11);\n"
        "  unixTempFileInit();\n"
        "  return 0;",
        "S1. Both call sqlite3_vfs_register exactly four times and then do "
        "tmpdir/mutex setup before returning 0; the register call dominates both.\n"
        "S2. Identical skeleton: four sequential register calls, a little init, a "
        "single return 0 — no loops or branches on either side.\n"
        "M1. Both are sqlite3_os_init: register the four unix VFS objects and "
        "initialize OS-level state.\n"
        "M2. B is the same whole top-level function as A (it just shows a couple of "
        "extra inlined init helpers), so the outer identity matches.",
        "HOST_MATCH",
    ),
    (
        "const char *__fastcall ldap_rdnfree(int *a1)\n"
        "  result = \" line %lu: changetype '%.*s' found but entries only was requested\\n\";\n"
        "  if ( a1 )\n"
        "    v2 = *a1;\n"
        "    if ( *a1 )\n"
        "      v4 = a1 + 1;\n"
        "      do\n"
        "        if ( (*(_BYTE *)(v2 + 19) & 0x20) != 0 )\n"
        "          ber_memfree_x(*(_DWORD *)(v2 + 12), 0);\n"
        "        ber_memfree_x(v2, 0);\n"
        "        v2 = *v4;\n"
        "      while ( *v4++ != 0 );\n"
        "    return (const char *)ber_memfree_x(a1, 0);\n"
        "  return result;",
        "void __cdecl ldap_rdnfree_x(LDAPRDN rdn, void *ctx)\n"
        "  if ( rdn )\n"
        "    v2 = *rdn;\n"
        "    if ( *rdn )\n"
        "      v3 = rdn + 1;\n"
        "      do\n"
        "        ++v3;\n"
        "        ldapava_free(v2);\n"
        "        v2 = *(v3 - 1);\n"
        "      while ( v2 );\n"
        "    ber_memfree_x();",
        "S1. Both walk an array of RDN entries calling ber_memfree_x; A carries an "
        "extra unrelated leading string literal that B does not.\n"
        "S2. The free-loop skeleton is the same (guard, take first entry, do-while "
        "over the array, final free), but A wraps it with extra host code.\n"
        "M1. B = ldap_rdnfree_x: free every attribute-value of one LDAP RDN; A is a "
        "larger host routine that contains exactly that free loop.\n"
        "M2. B's whole body appears as an inlined fragment inside the larger A, so "
        "this is a callee-level (inline) match, not a whole-function match.",
        "INLINE_MATCH",
    ),
    (
        "__int64 __fastcall sub_DB50(unsigned __int64 a1, __int64 a2, __int64 a3)\n"
        "  v33 = __readfsqword(0x28u);\n"
        "  result = a1;\n"
        "  if ( a3 > 0 )\n"
        "    v30 = 3988292384LL;\n"
        "    v8 = 1LL;\n"
        "    do  /* build identity matrix */\n"
        "      *(_QWORD *)v7 = v8; v7 += 8; v8 *= 2LL;\n"
        "    while ( v7 != &v32 );\n"
        "    for ( i = 3988292384LL; ; i = *(__int64 *)((char *)&v30 + v9) )\n"
        "      v11 = 0LL;\n"
        "      if ( i )\n"
        "        v12 = &v30;\n"
        "        do  /* <-- gf2_matrix_times inlined */\n"
        "          if ( (i & 1) != 0 ) v11 ^= *v12;\n"
        "          ++v12; i >>= 1;\n"
        "        while ( i );\n"
        "      *(_QWORD *)&v29[v9] = v11;\n"
        "    /* ... the same xor/shift bit-loop is inlined several more times to\n"
        "       square the matrix and apply it to v5 ... */\n"
        "      if ( (v4 & 1) != 0 && v5 )\n"
        "        v27 = v5; v28 = v29; v5 = 0LL;\n"
        "        do  /* <-- gf2_matrix_times inlined again */\n"
        "          if ( (v27 & 1) != 0 ) v5 ^= *v28;\n"
        "          ++v28; v27 >>= 1;\n"
        "        while ( v27 );\n"
        "      v4 >>= 2;\n"
        "    while ( v4 );\n"
        "    return a2 ^ v5;\n"
        "  return result;",
        "unsigned int __usercall gf2_matrix_times@<eax>(unsigned int *mat@<eax>, unsigned int vec@<edx>)\n"
        "  for ( i = 0; vec; vec >>= 1 )\n"
        "    if ( (vec & 1) != 0 )\n"
        "      i ^= *mat;\n"
        "    ++mat;\n"
        "  return i;",
        "S1. Both use the same GF(2) idiom — xor-accumulate selected matrix words while "
        "shifting a vector right; A also carries the CRC polynomial 3988292384 (0xEDB88320).\n"
        "S2. A is a large CRC-combine with matrix-build/square loops; B is one tight "
        "xor/shift bit-loop — and that exact loop appears verbatim SEVERAL times inside A.\n"
        "M1. B = gf2_matrix_times: multiply a GF(2) bit-matrix by a vector; A = crc32 "
        "combine, which repeatedly applies that matrix-times-vector operation.\n"
        "M2. B is tiny, but its distinctive bit-loop is inlined (more than once) inside the "
        "much larger A — a small inlined callee still counts as INLINE_MATCH.",
        "INLINE_MATCH",
    ),
]


def _assistant_text(reasoning, label):
    return reasoning.rstrip() + "\n\nANSWER: " + label


def build_messages(query, candidate):
    """Chat messages: system + few-shot (user/assistant CoT turns) + the real pair."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, c, reasoning, lab in FEW_SHOT:
        msgs.append({"role": "user", "content": PAIR_TMPL.format(q=q, c=c)})
        msgs.append({"role": "assistant", "content": _assistant_text(reasoning, lab)})
    msgs.append({"role": "user", "content": PAIR_TMPL.format(q=query, c=candidate)})
    return msgs


def parse_label(text):
    """Extract the discrete class from a CoT response -> 0/1/2 or None.

    Prefer the explicit `ANSWER: <LABEL>` line; fall back to the last label token
    that appears anywhere; finally fall back to a leading first-letter cue. Never
    returns a score — the caller maps the label to a match flag itself."""
    if not text:
        return None
    up = text.upper()
    idx = up.rfind("ANSWER:")
    if idx != -1:
        tail = up[idx + len("ANSWER:"): idx + len("ANSWER:") + 40]
        for li, lab in enumerate(LABELS):
            if lab in tail:
                return li
        c = tail.strip()[:1]
        m = {"N": 0, "H": 1, "I": 2}.get(c)
        if m is not None:
            return m
    best = (-1, None)
    for li, lab in enumerate(LABELS):
        p = up.rfind(lab)
        if p > best[0]:
            best = (p, li)
    return best[1]
