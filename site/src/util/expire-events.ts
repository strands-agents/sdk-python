// Build-time filtering is the primary gate; this is the client-side freshness backstop.

/** Remove expired `[data-expires]` elements, then hide/adjust ticker and poster. */
export function expireEvents(root: Document | Element, todayIso: string): void {
  root.querySelectorAll<HTMLElement>('[data-expires]').forEach((el) => {
    if (el.dataset.expires! < todayIso) el.remove()
  })

  const ticker = root.querySelector<HTMLElement>('.ticker')
  if (ticker && !ticker.querySelector('[data-expires]')) ticker.hidden = true

  const poster = root.querySelector<HTMLElement>('.poster')
  if (poster) {
    const list = poster.querySelector<HTMLElement>('.list')
    if (list) {
      if (!list.querySelector('[data-expires]')) {
        list.remove()
      } else if (!poster.querySelector('#poster-headliner')) {
        // list--bare removes border-top so the list doesn't float with a gap when headliner expired
        list.classList.add('list--bare')
      }
    }
    if (!poster.querySelector('[data-expires]')) {
      const evergreen = root.querySelector<HTMLElement>('#poster-evergreen')
      if (evergreen) evergreen.removeAttribute('hidden')
      const cal = root.querySelector<HTMLElement>('#poster-cal')
      if (cal) cal.textContent = 'Join the Discord →'
    }
  }
}
