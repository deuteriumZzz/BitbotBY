// Всё про Telegram в одном месте: бот и уведомления, вход в аккаунт и каналы для сигналов.
// В «Профи» — панель блока «Telegram» на схеме, в «Простом» — карточки в Настройках.

import { AlertsCard } from "./alerts";
import { KeyLine } from "./keys";
import { TgChannelsCard } from "./telegram";

export function TelegramPanel() {
  return (
    <>
      <AlertsCard />
      <article className="card">
        <KeyLine service="telethon" label="Вход в Telegram для чтения каналов" />
      </article>
      <TgChannelsCard />
    </>
  );
}
