function [LToe,LHeel,RToe,RHeel] = findFeet(img)

    [rows, columns, ~] = size(img);
    
    LToeVar = 5;
    RToeVar = -5;
    LHeelVar = -5;
    RHeelVar = 5;
    
    LToeFound = false;
    LHeelFound = false;
    RToeFound = false;
    RHeelFound = false; 
    
    %LToe & LHeel
    for j = 1:columns
        allWhite = true;
        for i = 1:rows
            
            %LToe
            if img(i,j) == 0 && LToeFound == false
                i0 = i;
                while img(i0,j+LToeVar) == 0
                   i0 = i0+1;
                end
                LToe = [i0,j+LToeVar];
                LToeFound = true;
            end
            
            if (img(i,j)==0)
                allWhite = false;
            end
            
            %LHeel
            if (i == rows && LToeFound == true && allWhite == true)
                for i0 = 1:rows
                    if img(i0,j-1+LHeelVar) == 0
                        LHeel = [i0,j-1+LHeelVar];
                        LHeelFound = true;
                        break
                    end
                end
                break
            end
        end
        
        if LHeelFound == true
            break
        end
    end

    %RToe & RHeel    
    for j = 1:columns
        allWhite = true;
        for i = 1:rows
            
            %RToe
            if img(i,columns-j) == 0 && RToeFound == false
                i0 = i;
                while img(i0,columns-j+RToeVar) == 0
                   i0 = i0+1;
                end
                RToe = [i0,columns-j+RToeVar];
                RToeFound = true;
            end
            
            if (img(i,columns-j)==0)
                allWhite = false;
            end
            
            %RHeel
            if (i == rows && RToeFound == true && allWhite == true)
                for i0 = 1:rows
                    if img(i0,columns-j+1+RHeelVar) == 0
                        RHeel = [i0,columns-j+1+RHeelVar];
                        RHeelFound = true;
                        break
                    end
                end
                break
            end
        end
        
        if RHeelFound == true
            break
        end
    end
    
end