import cv2
import numpy as np

# Constants for image processing
epsilon = 10  # Error sensitivity for corner detection
test_sensitivity_epsilon = 10  # Sensitivity for bubble darkness detection
answer_choices = ['A', 'B', 'C', 'D', 'E', '?']  # Possible answers

# Load tracking tags - These are your markers at the corners of the page
tags = [cv2.imread("markers/top_left.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/top_right.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)]

# Exam sheet scaling constants
scaling = [605.0, 835.0]  # Scaling factor for an 8.5in. x 11in. paper
columns = [[69.0 / scaling[0], 36 / scaling[1]],  # First column
           [180.0 / scaling[0], 36 / scaling[1]],  # Second column
           [290.0 / scaling[0], 36 / scaling[1]],  # Third column
           [403.0 / scaling[0], 36 / scaling[1]],  # Third column
           [515.0 / scaling[0], 36 / scaling[1]]]  # Fourth column
radius =  6.0 / scaling[0]  # Radius of the bubbles
spacing = [20.0 / scaling[0], 19.65 / scaling[1]]  # Spacing between bubbles

def ProcessPage(paper):
    answers = []  # Will contain the answers
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    corners = FindCorners(paper)  # Find the corners of the answer areas

    if corners is None:
        return [-1], paper  # Return error if corners are not found

    # Calculate the dimensions for scaling based on detected corners
    dimensions = [corners[1][0] - corners[0][0], corners[2][1] - corners[0][1]]

    # Iterate over each question
    for k in range(0, 5): #columns
        for i in range(0, 40): #rows
            questions = []
            for j in range(0, 4): #answers
                #coordinates of the answer bubble
                x1 = int((columns[k][0] + j*spacing[0] - radius*1.5)*dimensions[0] + corners[0][0])
                y1 = int((columns[k][1] + i*spacing[1] - radius)*dimensions[1] + corners[0][1])
                x2 = int((columns[k][0] + j*spacing[0] + radius*1.5)*dimensions[0] + corners[0][0])
                y2 = int((columns[k][1] + i*spacing[1] + radius)*dimensions[1] + corners[0][1])

                #draw rectangles around bubbles
                cv2.rectangle(paper, (x1, y1), (x2, y2), (255, 0, 0), thickness=1, lineType=8, shift=0)

                #crop answer bubble
                questions.append(gray_paper[y1:y2, x1:x2])

            #find image means of the answer bubbles
            means = []

            #coordinates to draw detected answer
            x1 = int((columns[k][0] - radius*8)*dimensions[0] + corners[0][0])
            y1 = int((columns[k][1] + i*spacing[1] + 0.5*radius)*dimensions[1] + corners[0][1])

            #calculate the image means for each bubble
            for question in questions:
                means.append(np.mean(question))

            #sort by minimum mean; sort by the darkest bubble
            min_arg = np.argmin(means)
            min_val = means[min_arg]

            #find the second smallest mean
            means[min_arg] = 255
            min_val2 = means[np.argmin(means)]

            #check if the two smallest values are close in value
            print("MIN: "+str(min_val))
            print("MIN2: "+str(min_val2))
            if min_val2 - min_val < test_sensitivity_epsilon:
                #if so, then the question has been double bubbled and is invalid
                min_arg = 5

            #write the answer
            cv2.putText(paper, answer_choices[min_arg], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)

            #append the answers to the array
            answers.append(answer_choices[min_arg])


    return answers, paper
# ... [other supporting functions like `FindCorners`] ...

def get_bubble_coords(columns, col_index, row, choice, dimensions, corners, radius, spacing):
    x1 = int((columns[col_index][0] + choice * spacing[0] - radius * 1.5) * dimensions[0] + corners[0][0])
    y1 = int((columns[col_index][1] + row * spacing[1] - radius) * dimensions[1] + corners[0][1])
    x2 = int((columns[col_index][0] + choice * spacing[0] + radius * 1.5) * dimensions[0] + corners[0][0])
    y2 = int((columns[col_index][1] + row * spacing[1] + radius) * dimensions[1] + corners[0][1])
    return x1, y1, x2, y2
def FindCorners(paper):
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)  # convert image of paper to grayscale

    # scaling factor used later
    ratio = len(paper[0]) / 816.0

    # error detection
    if ratio == 0:
        return None

    corners = []  # array to hold found corners

    # try to find the tags via convolving the image
    for tag in tags:
        tag = cv2.resize(tag, (0, 0), fx=ratio, fy=ratio)  # resize tags to the ratio of the image

        # convolve the image
        convimg = (cv2.filter2D(np.float32(cv2.bitwise_not(gray_paper)), -1, np.float32(cv2.bitwise_not(tag))))

        # find the maximum of the convolution
        corner = np.unravel_index(convimg.argmax(), convimg.shape)

        # append the coordinates of the corner
        corners.append([corner[1], corner[0]])  # reversed because array order is different than image coordinate

    # draw the rectangle around the detected markers
    for corner in corners:
        cv2.rectangle(paper, (corner[0] - int(ratio * 25), corner[1] - int(ratio * 25)),
                      (corner[0] + int(ratio * 25), corner[1] + int(ratio * 25)), (0, 255, 0), thickness=2, lineType=8, shift=0)

    # check if detected markers form roughly parallel lines when connected
    if corners[0][0] - corners[2][0] > epsilon or corners[1][0] - corners[3][0] > epsilon or corners[0][1] - corners[1][1] > epsilon or corners[2][1] - corners[3][1] > epsilon:
        return None

    return corners

# Example usage
# Load a test paper image
test_paper = cv2.imread("img_7.png")

# Process the test paper
answers, processed_paper = ProcessPage(test_paper)

# Display the processed paper
print(answers)
cv2.imshow("Processed Paper", processed_paper)
cv2.waitKey(0)
cv2.destroyAllWindows()
