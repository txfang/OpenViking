# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Search Service for OpenViking.

Provides search operations: search, find.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import asyncio
from openviking.core.namespace_policy import is_system_uri
from openviking.core.retrieval_types import SearchType
from openviking.server.identity import RequestContext
from openviking.storage.viking_fs import VikingFS
from openviking.utils.image_search import (
    image_bytes_to_data_uri,
    is_data_image_uri,
    is_http_url,
    is_viking_uri,
)
from openviking.utils.model_retry import model_retry_scope
from openviking_cli.exceptions import InvalidArgumentError, NotInitializedError
from openviking_cli.retrieve import ContextType
from openviking_cli.utils import get_logger

if TYPE_CHECKING:
    from openviking.session import Session

logger = get_logger(__name__)


def _reject_system_targets(target_uri: Union[str, List[str]]) -> None:
    targets = [target_uri] if isinstance(target_uri, str) else target_uri
    for uri in targets:
        if uri and is_system_uri(uri):
            raise InvalidArgumentError(f"search is not supported for namespace: {uri}")


def _validate_query(
    query: str,
    image_url: str | list[str] | None = None,
    filter: Optional[Dict] = None,
) -> None:
    """Reject malformed query text or a request with nothing to search for.

    A filter is an acceptable substitute for a query: the result set is then
    fully determined by the filter, which is exactly what an exact-match lookup
    (e.g. by a tag carrying an external id) needs. Without either one the call
    would return an arbitrary slice of the whole store.
    """
    try:
        query.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidArgumentError("Search query must contain valid Unicode text.") from exc
    if query.strip() or image_url or filter:
        return
    raise InvalidArgumentError(
        "Search query or image_url must not be empty unless a filter is provided."
    )


class SearchService:
    """Search service."""

    def __init__(self, viking_fs: Optional[VikingFS] = None):
        self._viking_fs = viking_fs

    def set_viking_fs(self, viking_fs: VikingFS) -> None:
        """Set VikingFS instance (for deferred initialization)."""
        self._viking_fs = viking_fs

    def _ensure_initialized(self) -> VikingFS:
        """Ensure VikingFS is initialized."""
        if not self._viking_fs:
            raise NotInitializedError("VikingFS")
        return self._viking_fs

    def is_intent_enabled(self) -> bool:
        """Whether search uses session context for LLM intent analysis.

        When false, callers should skip session.load / get_context_for_search:
        VikingFS.search ignores session_info and searches with the raw query.
        Default is True (matches RetrievalConfig) when config is unset.
        """
        if not self._viking_fs or self._viking_fs.retrieval_config is None:
            return True
        return bool(self._viking_fs.retrieval_config.enable_intent)

    async def ensure_keywords_search_supported(self, ctx: RequestContext) -> None:
        """Validate keyword-search support at the top-level request boundary."""
        viking_fs = self._ensure_initialized()
        vector_store = viking_fs._get_vector_store()
        if not vector_store or not await viking_fs._collection_has_fulltext(
            vector_store,
            ctx,
            supported_modes=("volcengine", "vikingdb"),
            raise_on_error=True,
        ):
            raise InvalidArgumentError(
                "Keyword search is not supported by the configured vector backend, or full-text "
                "indexing is not enabled for the content field."
            )

    async def _resolve_image_url(
        self,
        image_url: str | list[str] | None,
        ctx: RequestContext,
    ) -> str | list[str] | None:
        if isinstance(image_url, list):
            return [await self._resolve_image_url(url, ctx) for url in image_url]
        if not image_url:
            return None
        if is_viking_uri(image_url):
            viking_fs = self._ensure_initialized()
            content = await viking_fs.read_file_bytes(image_url, ctx=ctx)
            return image_bytes_to_data_uri(content, image_url)
        if is_data_image_uri(image_url) or is_http_url(image_url):
            return image_url
        raise InvalidArgumentError(
            "image_url must be a data:image base64 URI, http(s) URL, or viking:// URI."
        )

    @model_retry_scope(0)
    async def search(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]] = "",
        session: Optional["Session"] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
        level: Optional[List[int]] = None,
        image_url: str | list[str] | None = None,
        events_time_decay_protection: Optional[str] = None,
        search_type: SearchType = "semantic",
        context_types: Optional[List[ContextType]] = None,
    ) -> Any:
        """Search with session context.

        Args:
            query: Query string
            target_uri: Target directory URI(s), supports str or List[str]
            session: Session object for context
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters
            level: Filter by level (0=abstract, 1=overview, 2=file)

        Returns:
            FindResult
        """
        if search_type == "keywords" and image_url:
            raise InvalidArgumentError("image_url is not supported when search_type='keywords'")
        resolved_image_url = await self._resolve_image_url(image_url, ctx)
        _validate_query(query, resolved_image_url)
        _reject_system_targets(target_uri)
        if search_type == "keywords":
            await self.ensure_keywords_search_supported(ctx)
        viking_fs = self._ensure_initialized()

        session_info = None
        # Intent off: session_info is unused by VikingFS — skip the archive/message scan.
        if session is not None and self.is_intent_enabled() and not resolved_image_url:
            session_info = await session.get_context_for_search(query)

        result = await viking_fs.search(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            session_info=session_info,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
            level=level,
            image_url=resolved_image_url,
            events_time_decay_protection=events_time_decay_protection,
            search_type=search_type,
            context_types=context_types,
        )
        return result

    @model_retry_scope(0)
    async def find(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]] = "",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
        level: Optional[List[int]] = None,
        image_url: str | list[str] | None = None,
        events_time_decay_protection: Optional[str] = None,
        search_type: SearchType = "semantic",
    ) -> Any:
        """Search without session context.

        Args:
            query: Query string
            target_uri: Target directory URI(s), supports str or List[str]
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters
            level: Filter by level (0=abstract, 1=overview, 2=file)

        Returns:
            FindResult
        """
        if search_type == "keywords" and image_url:
            raise InvalidArgumentError("image_url is not supported when search_type='keywords'")
        resolved_image_url = await self._resolve_image_url(image_url, ctx)
        _validate_query(query, resolved_image_url, filter)
        _reject_system_targets(target_uri)
        viking_fs = self._ensure_initialized()
        result = await viking_fs.find(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
            level=level,
            image_url=resolved_image_url,
            events_time_decay_protection=events_time_decay_protection,
            search_type=search_type,
        )
        return result

    async def find_skills(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        level: Optional[List[int]] = None,
        filter: Optional[Dict] = None,
        search_type: SearchType = "semantic",
    ) -> Any:
        """Find distinct packages; general find/search stay item-based."""
        from openviking.core.retrieval_targets import resolve_retrieval_targets
        from openviking.retrieve.skill_package_retriever import SkillPackageRetriever
        from openviking.retrieve.skill_results import SkillResultResolver
        from openviking_cli.retrieve import ContextType, FindResult, TypedQuery

        _validate_query(query)
        _reject_system_targets(target_uri)
        fs = self._ensure_initialized()
        targets = resolve_retrieval_targets(target_uri, ctx).target_directories
        for target in targets:
            await fs._ensure_retrieval_scope(target, ctx)
        storage, embedder = fs._get_vector_store(), fs._get_embedder(ctx)
        if not storage:
            raise RuntimeError("Vector store not initialized. Call OpenViking.initialize() first.")
        if search_type == "semantic" and not embedder:
            raise RuntimeError("Embedder not configured.")
        retriever = SkillPackageRetriever(
            storage=storage,
            embedder=embedder,
            rerank_config=fs.rerank_config,
        )
        result = await retriever.retrieve_skills(
            TypedQuery(query, ContextType.SKILL, "", target_directories=targets),
            ctx,
            skill_resolver=SkillResultResolver(fs, ctx),
            limit=limit,
            score_threshold=score_threshold,
            level=level,
            scope_dsl=filter,
            search_type=search_type,
        )
        return FindResult(memories=[], resources=[], skills=result.matched_contexts)

    async def find_evidence(
        self,
        query: str,
        ctx: RequestContext,
        user_uri: str = "",
        top_k: int = 15,
        epi_window: int = 4,
        propagate: bool = True,
        use_semantic: bool = False,
        sem_anchor_top: int = 3,
        closure_radius: int = 1,
        strategy: str = "legacy",
        candidate_limit: int = 50,
        max_supplementary_queries: int = 2,
        use_memories: bool = False,
        use_events: bool = False,
        event_top_k: int = 2,
    ) -> Dict[str, Any]:
        """Zero-Mem/EdgeMem style evidence retrieval over archived raw turns.
        Reads the user-space evidence index (``{user_uri}/.index/evidence.json``)
        built incrementally at session commit, and returns ranked raw turns
        (episode/turn/speaker/text/date/uri) so the caller can assemble an
        answer prompt without a second read. Deterministic, zero LLM.
        A missing index yields an empty evidence list. Storage, authorization
        and parsing errors propagate rather than masquerading as an empty index.
        """
        if (use_memories or use_events) and strategy != "coverage":
            raise InvalidArgumentError("Memory and event channels require strategy=coverage")
        if not 1 <= event_top_k <= 20:
            raise InvalidArgumentError("event_top_k must be between 1 and 20")
        viking_fs = self._ensure_initialized()
        user_uri = (user_uri or f"viking://user/{ctx.user.user_id}").rstrip("/")
        index_uri = f"{user_uri}/.index/evidence.json"
        if strategy == "coverage":
            from openviking.retrieve.evidence_retriever import EvidenceIndex
            from openviking.retrieve.evidence_strategy import retrieve_coverage

            # Re-read through the filesystem's current request authorization.
            index = await EvidenceIndex.load(viking_fs, index_uri, ctx)

            async def embed(query_text):
                if not use_semantic:
                    return None
                from openviking_cli.utils.config import get_openviking_config
                embedder = get_openviking_config().embedding.get_embedder()
                if embedder is None:
                    return None
                return await asyncio.to_thread(self._embed_query, embedder, query_text)

            sources = None
            event_cap = 1.
            if use_memories or use_events:
                from openviking.retrieve.memory_sources import load_memory_sources, MemorySources
                from openviking.retrieve.evidence_strategy import configured_event_cap
                if use_events:
                    event_cap = configured_event_cap()
                try:
                    sources = await asyncio.wait_for(load_memory_sources(
                        viking_fs, index, user_uri, ctx,
                        use_memories=use_memories, use_events=use_events,
                    ), timeout=20.)
                except TimeoutError:
                    sources = MemorySources()
                    sources.diagnostics["errors"].append(dict(stage="memory_sources", error_type="TimeoutError"))

            evidence, diagnostics = await retrieve_coverage(
                index, query, embed=embed,
                top_k=top_k, candidate_limit=candidate_limit,
                max_supplementary_queries=max_supplementary_queries,
                memory_sources=sources, event_top_k=event_top_k, event_cap=event_cap,
            )
            return dict(query=query, index_uri=index_uri, n_evidence=len(evidence),
                        evidence=evidence,
                        facts=sources.selected_facts if sources is not None else [],
                        diagnostics=diagnostics)
        if strategy != "legacy":
            raise InvalidArgumentError(f"Unknown evidence strategy: {strategy}")
        from openviking.retrieve.evidence_retriever import load_index
        index = await load_index(viking_fs, index_uri, ctx)
        q_vec: Optional[List[float]] = None
        if use_semantic:
            try:
                from openviking_cli.utils.config import get_openviking_config
                embedder = get_openviking_config().embedding.get_embedder()
                if embedder is not None:
                    q_vec = await asyncio.to_thread(self._embed_query, embedder, query)
            except Exception as exc: # pragma: no cover - fail-open
                logger.warning("[evidence] semantic query embed failed: %s", exc)
                q_vec = None
        evidence = index.retrieve(
            query=query,
            top_k=top_k,
            epi_window=epi_window,
            propagate=propagate,
            use_semantic=use_semantic,
            q_vec=q_vec,
            sem_anchor_top=sem_anchor_top,
            closure_radius=closure_radius,
        )
        return {
            "query": query,
            "index_uri": index_uri,
            "n_evidence": len(evidence),
            "evidence": evidence,
        }
    @staticmethod
    def _embed_query(embedder: Any, text: str) -> Optional[List[float]]:
        """Synchronous query embedding -> dense vector list (fail-open)."""
        try:
            result = embedder.embed(text, is_query=True)
            vec = getattr(result, "dense_vector", None)
            return list(vec) if vec else None
        except Exception as exc: # pragma: no cover - defensive
            logger.warning("[evidence] embedder.embed failed: %s", exc)
        return None
