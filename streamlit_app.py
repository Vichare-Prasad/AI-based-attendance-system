pip install matplotlib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from datetime import datetime
import face_recognition
import os
import speech_recognition as sr

# Function to recognize speech
def recognize_present(timeout=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.write(f"Listening for 'present' for {timeout} seconds...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source, timeout=timeout)
    
    try:
        spoken_word = recognizer.recognize_google(audio_data)
        if "present" in spoken_word.lower():
            st.sidebar.success("Student is present.")
            return "present"
        else:
            st.sidebar.error("Did not recognize 'present'.")
            return None
    except sr.UnknownValueError:
        st.sidebar.error("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        st.sidebar.error(f"Could not request results from Google Speech Recognition service: {e}")
        return None

# Function to speak text
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to display real-time attendance dashboard
def attendance_dashboard(attendance_df, face_recog_record_df, show_class_attendance):
    st.title("Real-Time Attendance Dashboard")

    attendance_summary = attendance_df[attendance_df.columns[3:]].apply(lambda col: col.str.upper())
    attendance_summary = attendance_summary.apply(pd.Series.value_counts).transpose()
    present_counts = attendance_summary.get('P', pd.Series([0] * len(attendance_summary)))

    if show_class_attendance:
        # Analyze attendance trends for the whole class
        st.write("### Class Attendance Analysis")

        # Plot a bar graph for the whole class attendance
        fig, ax = plt.subplots()
        index = range(1, len(attendance_summary) + 1)
        bar_width = 0.35
        opacity = 0.8

        rects = ax.bar(index, present_counts, bar_width, alpha=opacity, color='b', label='Present')

        ax.set_xlabel('Roll Number')
        ax.set_ylabel('Number of Lectures')
        ax.set_title('Class Attendance')
        ax.set_xticks(index)
        ax.set_yticks(range(int(max(present_counts)) + 2))
        ax.legend()

        st.write("#### Attendance Graph for the Whole Class")
        st.pyplot(fig)
    else:
        # Display search bar for roll number
        roll_number_search = st.text_input("Enter roll number to search:")

        if roll_number_search:
            roll_number_search = int(roll_number_search)

            # Total number of lectures attended
            lectures_attended = present_counts[roll_number_search-1]

            # Attendance percentage
            total_lectures = len(attendance_df)
            attendance_percentage = (lectures_attended / total_lectures) * 100

            # Display total lectures attended and attendance percentage for the specified roll number
            st.write(f"### Roll Number: {roll_number_search}")
            st.write(f"Total Lectures Attended: {lectures_attended}")
            st.write(f"Attendance Percentage: {attendance_percentage:.2f}%")

            # Display all data for the specified roll number from face_recog_record.csv
            st.write("### Roll Number Data from face_recog_record.csv")
            roll_data = face_recog_record_df[face_recog_record_df["Roll Number"] == roll_number_search]
            st.write(roll_data)

# Streamlit app
def main():
    st.set_page_config(page_title="Integrated Attendance System", page_icon="📊")

    st.sidebar.title("Select Functionality")
    option = st.sidebar.radio("Choose:", ("Attendance System", "Voice Recognition Roll Call", "Attendance Dashboard"))

    if option == "Attendance System":
        st.title("Real-Time Face Recognition Attendance System")

        # Check Webcam
        st.header("Webcam Check")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to access webcam.")
                return
            else:
                st.success("Webcam is working properly.")
                if st.button("Take Picture"):
                    ret, frame = cap.read()
                    if ret:
                        # Save the captured image
                        image_path = "captured_image.jpg"
                        cv2.imwrite(image_path, frame)
                        st.success("Picture captured successfully!")
                        # Display the captured image
                        st.image(image_path)

                        # Load known students
                        known_students = {
                            "01": {"name": "Dhruve", "image_path": r"content\Dhruve.jpg"},
                            "02": {"name": "Prasad", "image_path": r"content\Prasad.jpg"},
                            "03": {"name": "Sakshi", "image_path": r"content\Sakshi.jpg"},
                            "04": {"name": "Diya", "image_path": r"content\Diya.jpg"},
                        }

                        stored_image = cv2.imread(image_path)

                        # Encode known students
                        encoded_students = {}
                        for roll_number, student_info in known_students.items():
                            student_image = face_recognition.load_image_file(student_info["image_path"])
                            encoded_students[roll_number] = face_recognition.face_encodings(student_image)[0]

                        # Perform face recognition on the stored image
                        face_locations = face_recognition.face_locations(stored_image)
                        face_encodings = face_recognition.face_encodings(stored_image, face_locations)

                        # Load existing attendance records or create new DataFrame
                        if os.path.exists("face_recog_records.csv"):
                            df = pd.read_csv("face_recog_records.csv")
                        else:
                            df = pd.DataFrame(columns=["Roll Number", "Name", "Date", "Time", "Attendance"])

                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(list(encoded_students.values()), face_encoding)
                            roll_number = ""
                            if True in matches:
                                match_index = matches.index(True)
                                roll_number = list(encoded_students.keys())[match_index]
                                current_date = datetime.now().strftime("%d-%m-%Y")
                                current_time = datetime.now().strftime("%H:%M:%S")
                                # Check if the entry already exists in the DataFrame
                                if not ((df["Roll Number"] == roll_number) & (df["Date"] == current_date)).any():
                                    df = df.append({"Roll Number": roll_number,
                                                    "Name": known_students[roll_number]["name"],
                                                    "Date": current_date,
                                                    "Time": current_time,
                                                    "Attendance": "Present"}, ignore_index=True)
                            else:
                                st.warning("Unknown face detected. Attendance not recorded.")

                        # Save the DataFrame to a CSV file
                        df.to_csv("face_recog_records.csv", index=False)
                    else:
                        st.error("Failed to capture picture.")
                        return
            cap.release()
        except Exception as e:
            st.error(f"Error: {e}")
            return

    elif option == "Voice Recognition Roll Call":
        st.title("Voice Recognition Roll Call")

        st.sidebar.header("Settings")
        num_lecture = st.sidebar.number_input("Lecture number of the subject for the day",
                                               min_value=1, max_value=10, value=1)

        # Sidebar
        st.sidebar.header("Settings")
        if st.sidebar.button("Start Roll Call"):
            if os.path.exists("attendance.csv"):
                attendance = pd.read_csv("attendance.csv")
            else:
                attendance = pd.DataFrame(columns=["Lec_num", "Date", "Time", "1", "2", "3", "4"])

            if num_lecture in attendance['Lec_num'].values:
                st.sidebar.error("Roll Call already done for this lecture.")
                return

            st.sidebar.success("Roll Call Started...")

            enrolled_students = {"1": "Dhruve", "2": "Prasad", "3": "Sakshi", "4": "Diya"}

            face_recog = pd.read_csv('face_recog_records.csv')

            current_date = datetime.now().strftime("%d-%m-%Y")
            current_time = datetime.now().strftime("%H:%M:%S")

            attendance = attendance.append({'Lec_num': num_lecture,
                                            'Date': current_date,
                                            'Time': current_time,
                                            '1': 'A',
                                            '2': 'A',
                                            '3': 'A',
                                            '4': 'A',
                                            }, ignore_index=True)

            roll_call_output = []  # List to store roll call results

            for roll_number in enrolled_students.keys():
                st.sidebar.write(f"### Roll Number: {roll_number}")
                speak(f"Roll Number {roll_number}")
                student_name = enrolled_students[roll_number]
                st.write(f"Calling {student_name}...")
                st.sidebar.write("Please say 'present' when called.")
                st.sidebar.write("Listening...")
                status = recognize_present(timeout=5)
                if status:
                    # update_attendance(roll_number, status, roll_call_df)
                    roll_call_output.append(f"{student_name}: present")
                    st.success("Attendance marked.")

                    for index, row in face_recog.iterrows():
                        if int(roll_number) == row[0] and current_date == row[2]:
                            attendance.loc[(attendance['Date'] == current_date)
                                           & (attendance['Lec_num'] == num_lecture), roll_number] = "P"
                            attendance.to_csv('attendance.csv', index=False)
                else:
                    roll_call_output.append(f"{student_name}: not present")
                    st.error("Attendance not marked.")

            # Display the entire roll call process
            st.sidebar.header("Roll Call Summary")
            for output in roll_call_output:
                st.sidebar.write(output)

            st.sidebar.success("Roll Call Completed.")
            empty_roll_call = pd.DataFrame(columns=["Roll Number", "Attendance"])
            empty_roll_call.to_csv('Roll_call.csv', index=False)

    elif option == "Attendance Dashboard":
        st.title("Real-Time Attendance Dashboard")

        st.sidebar.header("Dashboard Settings")
        uploaded_attendance_file = st.sidebar.file_uploader("Upload attendance data (CSV)", type="csv")
        show_class_attendance = st.sidebar.radio("Show:", ("Class Attendance", "Individual Roll Number"))

        # Load attendance data
        if uploaded_attendance_file is not None:
            attendance_df = pd.read_csv(uploaded_attendance_file)
            face_recog_record_df = pd.read_csv("face_recog_records.csv")
            attendance_dashboard(attendance_df, face_recog_record_df, show_class_attendance == "Class Attendance")

if __name__ == "__main__":
    main()
