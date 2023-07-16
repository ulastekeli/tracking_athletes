#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_PNG_Image.H>

// Callback for when the button is clicked
void button_cb(Fl_Widget* btn, void* userdata) {
    Fl_Native_File_Chooser chooser;
    chooser.title("Pick a file");
    chooser.type(Fl_Native_File_Chooser::BROWSE_FILE);
    chooser.filter("Video\t*.{avi,mkv,mp4}");
    if (chooser.show() == 0) {
        // File was chosen
        const char* filename = chooser.filename();
        // Do something with the filename
    }
}

int main(int argc, char** argv) {
    Fl_Window* window = new Fl_Window(400, 300);
    
    Fl_Button* button = new Fl_Button(50, 50, 100, 25, "Choose File");
    button->callback(button_cb);
    
    Fl_Box* box = new Fl_Box(50, 100, 200, 200);
    Fl_PNG_Image* img = new Fl_PNG_Image("../data/example.jpg");
    box->image(img);
    
    window->end();
    window->show(argc, argv);
    return Fl::run();
}
