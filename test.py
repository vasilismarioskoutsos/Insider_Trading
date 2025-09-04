from datetime import date

def is_valid_day(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)   # raises ValueError if invalid
        return True
    except ValueError:
        return False

# examples
print(is_valid_day(2024, 2, 29))  # True  (leap year)
print(is_valid_day(2025, 2, 29))  # False
print(is_valid_day(2025, 4, 31))  # False