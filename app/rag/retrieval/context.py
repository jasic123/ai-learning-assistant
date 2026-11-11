"""Utilities for inspecting neighboring chunks around retrieval hits."""


def expand_neighbor_chunks(hits, chunks, *, window=1, page_window=0, boundary_chunks=3):
    """Return hit chunks plus same-page and bounded adjacent-page context."""
    if type(window) is not int or window < 0 or type(page_window) is not int or page_window < 0:
        raise ValueError("context windows must be non-negative integers")
    if type(boundary_chunks) is not int or boundary_chunks < 1:
        raise ValueError("boundary_chunks must be a positive integer")

    groups, source_pages, by_id = {}, {}, {}
    for chunk in chunks:
        groups.setdefault((chunk.source, chunk.page), []).append(chunk)
        source_pages.setdefault(chunk.source, set()).add(chunk.page)
        by_id[chunk.id] = chunk
    for page_chunks in groups.values():
        page_chunks.sort(key=lambda item: (item.offset, item.id))

    expanded, seen = [], set()
    for hit in hits:
        anchor = hit.chunk
        page_chunks = groups.get((anchor.source, anchor.page), ())
        anchor_index = next((i for i, chunk in enumerate(page_chunks)
                             if chunk.id == anchor.id), None)
        if window and (anchor.prev_chunk_id or anchor.next_chunk_id):
            before, after = [], []
            current = anchor
            for _ in range(window):
                current = by_id.get(current.prev_chunk_id)
                if current is None:
                    break
                before.append(current)
            current = anchor
            for _ in range(window):
                current = by_id.get(current.next_chunk_id)
                if current is None:
                    break
                after.append(current)
            candidates = list(reversed(before)) + [anchor] + after
        elif anchor_index is None:
            candidates = [anchor]
        else:
            start = max(0, anchor_index - window)
            end = min(len(page_chunks), anchor_index + window + 1)
            candidates = page_chunks[start:end]
        if page_window:
            for page in sorted(source_pages.get(anchor.source, ())):
                if 0 < abs(page - anchor.page) <= page_window:
                    page_chunks = groups[(anchor.source, page)]
                    candidates.extend(page_chunks[-boundary_chunks:] if page < anchor.page
                                      else page_chunks[:boundary_chunks])
            candidates.sort(key=lambda item: (item.page, item.offset, item.id))
        for chunk in candidates:
            if chunk.id not in seen:
                expanded.append(chunk)
                seen.add(chunk.id)
    return expanded
