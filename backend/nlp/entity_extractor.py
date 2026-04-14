import os
import warnings
from typing import List
from dataclasses import dataclass, field
from itertools import combinations

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


@dataclass
class Entity:
    name: str
    type: str
    span: tuple = (0, 0)
    confidence: float = 1.0

    @property
    def canonical_name(self) -> str:
        return self.name.lower().strip()


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str


@dataclass
class ExtractionResult:
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)


class EntityExtractor:
    def __init__(self):
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                try:
                    self._nlp = spacy.load("en_core_web_trf")
                except OSError:
                    raise OSError(
                        "No spaCy model found. Run: python -m spacy download en_core_web_sm"
                    )
        return self._nlp

    def _extract_dep_relations(self, doc) -> List[Relation]:
        """Extract subject-verb-object triples using spaCy dependency parse."""
        relations = []
        ent_spans = {ent.text.lower(): ent for ent in doc.ents}

        for token in doc:
            if token.pos_ != "VERB":
                continue
            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objects = [c for c in token.children if c.dep_ in ("dobj", "pobj", "attr", "prep")]
            for subj in subjects:
                subj_text = subj.text.lower()
                for obj in objects:
                    obj_text = obj.text.lower()
                    if subj_text != obj_text:
                        relations.append(Relation(
                            subject=subj_text,
                            predicate=token.lemma_.lower(),
                            object=obj_text,
                        ))
        return relations

    def extract(self, text: str) -> ExtractionResult:
        nlp = self._get_nlp()
        doc = nlp(text[:5000])
        entities = [
            Entity(name=ent.text, type=ent.label_, span=(ent.start_char, ent.end_char))
            for ent in doc.ents
        ]
        relations = self._extract_dep_relations(doc)
        # Add co-occurrence relations for all entity pairs in same text
        ent_names = [e.canonical_name for e in entities]
        for a, b in combinations(set(ent_names), 2):
            relations.append(Relation(subject=a, predicate="co_occurs_with", object=b))
        return ExtractionResult(entities=entities, relations=relations)

    def extract_batch(self, texts: list) -> list:
        nlp = self._get_nlp()
        results = []
        capped = [t[:5000] for t in texts]
        for doc in nlp.pipe(capped, batch_size=16):
            entities = [
                Entity(name=ent.text, type=ent.label_, span=(ent.start_char, ent.end_char))
                for ent in doc.ents
            ]
            relations = self._extract_dep_relations(doc)
            ent_names = [e.canonical_name for e in entities]
            for a, b in combinations(set(ent_names), 2):
                relations.append(Relation(subject=a, predicate="co_occurs_with", object=b))
            results.append(ExtractionResult(entities=entities, relations=relations))
        return results
