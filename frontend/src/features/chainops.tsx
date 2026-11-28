// Общие части раздела AI: статусы провайдера словами, время и значок.

import { aiLogo } from "../components/brand";

export const STATUS: Record<string, [string, string]> = {
  ok: ["ok", "работает"],
  rate_limit: ["warn", "лимит"],
  no_balance: ["bad", "нет денег"],
  bad_key: ["bad", "ключ не принят"],
  bad_model: ["bad", "нет такой модели"],
  bad_format: ["warn", "ломает формат ответа"],
  bad_request: ["warn", "отклонил запрос"],
  error: ["warn", "нет связи"],
  unknown: ["", "ещё не отвечал"],
};

export const hhmm = (epoch: number) =>
  new Date(epoch * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });

/** Значок провайдера: его логотип (по просьбе владельца — сразу видно, на ком работает бот); нет логотипа — монограмма. */
export function Avatar({ id, name }: { id: string; name: string }) {
  const logo = aiLogo(id, 20);
  if (logo)
    return (
      <span className="pav logo" aria-hidden="true">
        {logo}
      </span>
    );
  const h = [...id].reduce((acc, c) => (acc * 31 + c.charCodeAt(0)) % 360, 7);
  return (
    <span className="pav" style={{ background: `hsl(${h} 30% 22%)`, color: `hsl(${h} 65% 78%)`, borderColor: `hsl(${h} 32% 36%)` }} aria-hidden="true">
      {name.replace(/[^A-Za-zА-Яа-я0-9]/g, "").slice(0, 2)}
    </span>
  );
}

