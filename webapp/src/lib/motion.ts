// Общий слой движения: кромочный свет карточек.
// Граница карточки подсвечивается там, где курсор: переменные --mx/--my на карточке читает .card::after.
// Один слушатель на документ, не чаще кадра; на сенсорных экранах и при «меньше движения» не ставится.

export function installCardLight(): void {
  if (typeof window === "undefined" || typeof matchMedia !== "function") return;
  if (matchMedia("(prefers-reduced-motion: reduce)").matches || !matchMedia("(pointer: fine)").matches) return;
  let frame = 0;
  let last: PointerEvent | null = null;
  const paint = () => {
    frame = 0;
    const card = last && (last.target as Element | null)?.closest?.(".card");
    if (!(card instanceof HTMLElement) || !last) return;
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${last.clientX - r.left}px`);
    card.style.setProperty("--my", `${last.clientY - r.top}px`);
  };
  document.addEventListener(
    "pointermove",
    (e) => {
      last = e;
      if (!frame) frame = requestAnimationFrame(paint);
    },
    { passive: true },
  );
}
