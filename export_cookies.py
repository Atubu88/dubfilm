from browser_cookie3 import chrome

print("📥 Извлекаю куки из браузера...")

# Извлекаем cookies YouTube из Chrome
cookies = chrome(domain_name="youtube.com")

# Сохраняем в cookies.txt
with open("cookies.txt", "w", encoding="utf-8") as f:
    for c in cookies:
        f.write(
            f"{c.domain}\tTRUE\t{c.path}\t"
            f"{'TRUE' if c.secure else 'FALSE'}\t"
            f"{int(c.expires or 0)}\t"
            f"{c.name}\t{c.value}\n"
        )

print("✅ cookies.txt сохранён!")
print("➡ Загрузи этот файл в Render → Environment → Add Secret File")
