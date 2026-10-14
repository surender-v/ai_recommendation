import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
NUM_STUDENTS = 400

DEPARTMENTS = ['Computer Science', 'Electronics', 'Mechanical', 'Civil', 'Information Technology']
GENDERS     = ['Male', 'Female']
GUARDIANS   = ['Father', 'Mother', 'Guardian']
RESIDENCES  = ['Hostel', 'Day Scholar']
FEE_STATUS  = ['Paid', 'Partially Paid', 'Not Paid']
INCOME_BAND = ['Low', 'Medium', 'High']

FIRST_NAMES_M = ['Arjun', 'Karthik', 'Rahul', 'Santhosh', 'Vikram', 'Arun', 'Deepak', 'Gopal',
                  'Harish', 'Ilavarasan', 'Jagan', 'Kiran', 'Lokesh', 'Manoj', 'Naveen',
                  'Prasath', 'Ramesh', 'Senthil', 'Tamizh', 'Udhay']

FIRST_NAMES_F = ['Priya', 'Deepa', 'Kavya', 'Nithya', 'Sowmya', 'Anitha', 'Bhavani', 'Chitra',
                  'Divya', 'Eswari', 'Gayathri', 'Hema', 'Indhu', 'Janani', 'Keerthana',
                  'Lavanya', 'Meena', 'Nandhini', 'Oviya', 'Padmavathi']

LAST_NAMES = ['Kumar', 'Raj', 'Murugan', 'Krishnan', 'Rajan', 'Subramaniam', 'Pandian',
               'Balasubramanian', 'Selvan', 'Natarajan', 'Shanmugam', 'Venkatesan',
               'Annamalai', 'Durai', 'Periasamy']

# ─── HELPER: Generate realistic correlated marks ──────────────────────────────
# Students are categorised into 3 ability bands to make the dataset realistic
# Weak  → tends to score low,  Average → mid range,  Strong → scores high
def generate_student_marks(ability):
    """
    ability: 'weak' | 'average' | 'strong'
    CAT1  : out of 50
    CAT2  : out of 50
    Model : out of 100
    Returns (cat1, cat2, model)
    """
    if ability == 'weak':
        cat1  = np.clip(np.random.normal(20, 7),  5, 40)
        cat2  = np.clip(np.random.normal(18, 7),  3, 38)
        model = np.clip(np.random.normal(38, 12), 10, 75)
    elif ability == 'average':
        cat1  = np.clip(np.random.normal(33, 6),  15, 46)
        cat2  = np.clip(np.random.normal(31, 6),  15, 46)
        model = np.clip(np.random.normal(58, 10), 30, 85)
    else:  # strong
        cat1  = np.clip(np.random.normal(43, 4),  33, 50)
        cat2  = np.clip(np.random.normal(42, 4),  33, 50)
        model = np.clip(np.random.normal(80, 8),  60, 100)

    return round(cat1, 1), round(cat2, 1), round(model, 1)


# ─── GENERATE ROWS ────────────────────────────────────────────────────────────
rows = []

for i in range(1, NUM_STUDENTS + 1):
    # Basic demographics
    gender  = random.choice(GENDERS)
    if gender == 'Male':
        name = f"{random.choice(FIRST_NAMES_M)} {random.choice(LAST_NAMES)}"
    else:
        name = f"{random.choice(FIRST_NAMES_F)} {random.choice(LAST_NAMES)}"

    age        = random.randint(18, 23)
    dept       = random.choice(DEPARTMENTS)
    residence  = random.choice(RESIDENCES)
    guardian   = random.choice(GUARDIANS)
    income     = random.choice(INCOME_BAND)

    # Ability band distribution:  25 % weak | 50 % average | 25 % strong
    ability = random.choices(['weak', 'average', 'strong'], weights=[25, 50, 25])[0]

    cat1, cat2, model = generate_student_marks(ability)

    # Average Marks – normalise all three to 100 then average
    # CAT1 and CAT2 are out of 50 → multiply by 2 to bring to /100
    avg_marks = round(((cat1 * 2) + (cat2 * 2) + model) / 3, 2)

    # Attendance is correlated with ability
    if ability == 'weak':
        attendance = round(np.clip(np.random.normal(62, 10), 30, 85), 1)
    elif ability == 'average':
        attendance = round(np.clip(np.random.normal(78, 8),  55, 95), 1)
    else:
        attendance = round(np.clip(np.random.normal(89, 5),  75, 100), 1)

    # Fee status: low-income students more likely to have unpaid fees
    if income == 'Low':
        fee = random.choices(FEE_STATUS, weights=[30, 40, 30])[0]
    elif income == 'Medium':
        fee = random.choices(FEE_STATUS, weights=[60, 30, 10])[0]
    else:
        fee = random.choices(FEE_STATUS, weights=[85, 12,  3])[0]

    # Previous arrears (backlogs) correlated with ability
    if ability == 'weak':
        arrears = random.choices([0, 1, 2, 3, 4], weights=[20, 30, 25, 15, 10])[0]
    elif ability == 'average':
        arrears = random.choices([0, 1, 2, 3, 4], weights=[55, 25, 12,  6,  2])[0]
    else:
        arrears = random.choices([0, 1, 2, 3, 4], weights=[85, 10,  4,  1,  0])[0]

    # Internet / Study Access at Home
    internet_access = random.choices(['Yes', 'No'], weights=[70, 30])[0]

    rows.append({
        'Student_ID'          : f"S{i:04d}",
        'Name'                : name,
        'Age'                 : age,
        'Gender'              : gender,
        'Department'          : dept,
        'Residence'           : residence,
        'Guardian'            : guardian,
        'Family_Income'       : income,
        'CAT1_Marks'          : cat1,          # out of 50
        'CAT2_Marks'          : cat2,          # out of 50
        'Model_Exam_Marks'    : model,         # out of 100
        'Average_Marks'       : avg_marks,     # normalised average out of 100
        'Attendance_Percent'  : attendance,    # percentage
        'Previous_Arrears'    : arrears,       # number of backlogs
        'Fees_Status'         : fee,           # Paid / Partially Paid / Not Paid
        'Internet_Access'     : internet_access
    })

df = pd.DataFrame(rows)

# ─── SAVE ─────────────────────────────────────────────────────────────────────
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.abspath(os.path.join(current_dir, '..'))
data_dir    = os.path.join(root_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

output_file = os.path.join(data_dir, 'student_dataset.csv')
df.to_csv(output_file, index=False)

print(f"Dataset created successfully! -> {output_file}")
print(f"Total rows : {len(df)}")
print(f"\nColumn list:")
for col in df.columns:
    print(f"  • {col}")
print(f"\nSample data (first 5 rows):")
print(df.head().to_string(index=False))
print(f"\nBasic Stats:")
print(df[['CAT1_Marks','CAT2_Marks','Model_Exam_Marks','Average_Marks','Attendance_Percent']].describe().round(2))
print(f"\nFees_Status distribution:")
print(df['Fees_Status'].value_counts())
print(f"\nFamily_Income distribution:")
print(df['Family_Income'].value_counts())
