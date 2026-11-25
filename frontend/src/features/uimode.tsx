// Переключатель «Простой / Профи»: в шапке на широком экране, вверху вкладки «Бот» на узком.
// Это вид интерфейса на этом устройстве — настройки бота он не меняет.

import { useStore } from "../store";
import { Seg } from "../ui";

export function UiSwitch() {
  const ui = useStore((s) => s.ui);
  const setUi = useStore((s) => s.setUi);
  return (
    <Seg
      className="uimode"
      value={ui}
      options={[
        ["simple", "Простой"],
        ["pro", "Профи"],
      ]}
      onChange={setUi}
      label="Режим интерфейса"
    />
  );
}

/** Для узкого экрана: та же кнопка вверху вкладки, в шапке ей нет места. */
export function UiSwitchRow() {
  return (
    <div className="view-mode">
      <UiSwitch />
    </div>
  );
}
